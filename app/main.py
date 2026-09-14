"""FastAPI backend for the Jules Verne Bot web demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vernebot import DEFAULT_NUM_GENERATE, DEFAULT_TEMPERATURE, reflow
from vernebot.registry import ModelRegistry, build_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "static"

# Limits keep a demo request responsive and prevent runaway generation.
MAX_GENERATE = 4000
# Transformers are ~100x slower per token than the RNN in pure NumPy, so a
# 4000-character run would take over two minutes. Cap transformer runs lower by
# default; the client is told the real limit via /api/meta.
MAX_GENERATE_TRANSFORMER = 1500


class GenerateRequest(BaseModel):
    """A single text-generation request from the browser."""

    seed: str = Field(..., min_length=1, max_length=400, description="Seed word or phrase")
    model: str | None = Field(
        None, description="Model key from /api/models. Defaults to the first model."
    )
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.1, le=2.0)
    num_generate: int = Field(DEFAULT_NUM_GENERATE, ge=1, le=MAX_GENERATE)
    top_k: int | None = Field(None, ge=1, le=123)
    greedy: bool = False
    seed_value: int | None = Field(None, description="Optional RNG seed for reproducibility")
    reflow: bool = Field(
        True,
        description=(
            "Rejoin the hard-wrapped lines the model reproduces from the "
            "Gutenberg editions. Set false to see the raw character stream."
        ),
    )


class GenerateResponse(BaseModel):
    text: str
    seed: str
    model: str
    architecture: str
    temperature: float
    num_generate: int
    top_k: int | None
    greedy: bool
    seed_value: int | None
    reflow: bool
    elapsed_ms: float


class ModelSummary(BaseModel):
    key: str
    label: str
    blurb: str
    architecture: str
    model_file: str
    relative_path: str
    vocab_size: int | None = None
    parameter_count: int | None = None
    context_length: int | None = None
    training_loss: float | None = None
    max_generate: int = MAX_GENERATE
    loaded: bool = False


class CompareRequest(GenerateRequest):
    """Same controls as a normal run, plus the models to run them on."""

    models: list[str] = Field(..., min_length=1, max_length=4)


class CompareResult(BaseModel):
    model: str
    label: str
    architecture: str
    parameter_count: int | None = None
    context_length: int | None = None
    temperature: float
    text: str | None = None
    elapsed_ms: float | None = None
    error: str | None = None


class CompareResponse(BaseModel):
    seed: str
    temperature: float
    num_generate: int
    results: list[CompareResult]


class MetaResponse(BaseModel):
    default_model: str
    default_temperature: float
    default_num_generate: int
    max_generate: int
    models: list[ModelSummary]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Discover models at startup, loading weights lazily on first use.

    Lazy loading matters once several models are present: the registry starts in
    milliseconds and only reads the weights of the model actually used.
    """
    registry = build_registry(os.environ.get("VERNE_MODELS_DIR"))
    app.state.registry = registry
    app.state.models = registry.available()

    keys = registry.keys()
    if not keys:
        raise RuntimeError(
            "No models found in models/. Add verne_rnn_model.keras, or a "
            "Transformer .h5 exported by train_transformer.py."
        )
    # Loading the default now means the first request is fast.
    app.state.default_model = keys[0]
    registry.get(app.state.default_model)

    yield
    app.state.registry = None


app = FastAPI(
    title="Jules Verne Bot",
    description="Character-level text generation in the style of Jules Verne: "
                "a GRU RNN and decoder-only Transformers, served side by side.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def no_store_for_app_files(request, call_next):
    """Stop the browser caching the page, CSS and JS.

    Without this, edits to the front end do not appear until the user
    force-reloads, which is confusing while preparing a class. The API
    responses are POSTs and are not cached anyway, and serving these few
    local files afresh costs nothing.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


def get_registry() -> ModelRegistry:
    registry = getattr(app.state, "registry", None)
    if registry is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=503, detail="Model registry is not available.")
    return registry


def resolve(requested: str | None):
    """Look up a model entry, with a clear 404 for an unknown key."""
    registry = get_registry()
    key = requested or app.state.default_model
    try:
        return registry.get(key)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model {key!r}. Available: {', '.join(registry.keys())}",
        ) from exc


def summarise(entry, loaded: bool) -> ModelSummary:
    limit = (MAX_GENERATE_TRANSFORMER if entry.architecture == "transformer"
             else MAX_GENERATE)
    summary = ModelSummary(
        key=entry.key, label=entry.label, blurb=entry.blurb,
        architecture=entry.architecture, model_file=entry.model_file,
        relative_path=str(entry.path.relative_to(PROJECT_ROOT)),
        max_generate=limit, loaded=loaded,
    )
    if loaded:
        summary.vocab_size = entry.vocab_size
        summary.parameter_count = entry.parameter_count
        summary.context_length = entry.context_length
        summary.training_loss = entry.training_loss
    return summary


@app.get("/api/models", response_model=list[ModelSummary])
def list_models() -> list[ModelSummary]:
    """List available models. Weights are only read for models already loaded."""
    registry = get_registry()
    loaded = set(getattr(registry, "_loaded", {}))
    out = []
    for stub in registry.available():
        if stub["key"] in loaded:
            out.append(summarise(registry.get(stub["key"]), True))
        else:
            # Not loaded yet: report file-level facts only, so listing is instant.
            out.append(ModelSummary(
                key=stub["key"], label=stub["label"], blurb=stub["blurb"],
                architecture=stub["architecture"], model_file=stub["model_file"],
                relative_path=stub["relative_path"],
                max_generate=stub["max_generate"],
            ))
    return out


@app.get("/api/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    """Defaults and the model list, used to build the UI."""
    registry = get_registry()
    loaded = set(getattr(registry, "_loaded", {}))
    models = []
    for key in registry.keys():
        entry = registry.get(key)
        models.append(summarise(entry, key in loaded))

    return MetaResponse(
        default_model=app.state.default_model,
        default_temperature=DEFAULT_TEMPERATURE,
        default_num_generate=DEFAULT_NUM_GENERATE,
        max_generate=MAX_GENERATE,
        models=models,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text from a seed word, with temperature control.

    ``model`` selects the architecture, so the same request can be replayed
    against the RNN and a Transformer for a side-by-side comparison.
    """
    import time

    entry = resolve(request.model)
    model = entry.model

    # Drop characters the model never saw so the seed cannot break the lookup.
    cleaned_seed = "".join(c for c in request.seed if c in model.char_to_idx)
    if not cleaned_seed:
        bad = "".join(sorted(set(request.seed) - set(model.char_to_idx)))
        raise HTTPException(
            status_code=400,
            detail=(
                "The seed contains no characters from the model's vocabulary"
                + (f" (unsupported: {bad!r})" if bad else "")
                + ". Try plain English letters, digits or punctuation."
            ),
        )

    limit = (MAX_GENERATE_TRANSFORMER if entry.architecture == "transformer"
             else MAX_GENERATE)
    if request.num_generate > limit:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{entry.label} can generate at most {limit} characters per request. "
                "Transformers are far slower per character in pure NumPy; ask for "
                "less text, or use the RNN."
            ),
        )

    rng = np.random.default_rng(request.seed_value)
    started = time.perf_counter()
    text = "".join(
        model.generate(
            cleaned_seed,
            num_generate=request.num_generate,
            temperature=request.temperature,
            top_k=request.top_k,
            greedy=request.greedy,
            rng=rng,
        )
    )
    # The models reproduce the Gutenberg hard wrapping; rejoin it for display.
    if request.reflow:
        text = reflow(text)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return GenerateResponse(
        text=text,
        seed=cleaned_seed,
        model=entry.key,
        architecture=entry.architecture,
        temperature=request.temperature,
        num_generate=request.num_generate,
        top_k=request.top_k,
        greedy=request.greedy,
        seed_value=request.seed_value,
        reflow=request.reflow,
        elapsed_ms=round(elapsed_ms, 1),
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    """Run the same prompt across several models.

    Both models share the seed, the temperature and the sampler, so any
    difference in the writing comes from the weights and the architecture. One
    model failing (for example a too-long request against a slower Transformer)
    is reported in its own result rather than failing the whole comparison.
    """
    import time

    registry = get_registry()
    results: list[CompareResult] = []
    cleaned_for_seed: str | None = None

    for key in request.models:
        try:
            entry = resolve(key)
        except HTTPException as exc:
            results.append(CompareResult(
                model=key, label=key, architecture="unknown",
                temperature=request.temperature, error=str(exc.detail)))
            continue

        try:
            cleaned_seed = "".join(c for c in request.seed
                                   if c in entry.model.char_to_idx)
            if not cleaned_seed:
                raise ValueError("Seed contains no in-vocabulary characters.")
            cleaned_for_seed = cleaned_seed

            limit = (MAX_GENERATE_TRANSFORMER if entry.architecture == "transformer"
                     else MAX_GENERATE)
            if request.num_generate > limit:
                raise ValueError(
                    f"limit for this model is {limit} characters")

            rng = np.random.default_rng(request.seed_value)
            started = time.perf_counter()
            text = "".join(entry.model.generate(
                cleaned_seed, num_generate=request.num_generate,
                temperature=request.temperature, top_k=request.top_k,
                greedy=request.greedy, rng=rng))
            elapsed = (time.perf_counter() - started) * 1000
            if request.reflow:
                text = reflow(text)

            results.append(CompareResult(
                model=entry.key, label=entry.label, architecture=entry.architecture,
                parameter_count=entry.parameter_count,
                context_length=entry.context_length,
                temperature=request.temperature,
                text=text, elapsed_ms=round(elapsed, 1)))
        except (ValueError, KeyError) as exc:
            results.append(CompareResult(
                model=entry.key, label=entry.label, architecture=entry.architecture,
                parameter_count=entry.parameter_count,
                context_length=entry.context_length,
                temperature=request.temperature, error=str(exc)))

    return CompareResponse(
        seed=cleaned_for_seed or request.seed,
        temperature=request.temperature,
        num_generate=request.num_generate,
        results=results,
    )


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_file)
