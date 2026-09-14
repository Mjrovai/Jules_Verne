"""
Model registry: serve the RNN and any number of Transformers through one interface.

The web app should not care which architecture backs a model. This module
discovers the model files in ``models/``, wraps each in a small adapter with a
uniform surface, and hands the app an object that behaves like the original
``VerneRNN`` alone did.

Adding a model is deliberately boring: drop a file in ``models/`` and restart.
* ``*.keras``            -> the character-level GRU RNN
* ``*.h5``               -> a Transformer exported by ``train_transformer.py``

Each adapter exposes:

    key, label, architecture, vocab_size, parameter_count, context_length,
    model_file, training_loss      -> for the UI and for /api/meta
    char_to_idx                    -> for sanitising the seed
    generate(seed, ...)            -> the same generator interface as the RNN

That uniformity is what makes the side-by-side comparison honest: both models
receive the same seed, the same temperature and the same sampler, so any
difference in the output comes from the weights and architecture, not from the
plumbing around them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .engine import DEFAULT_MODEL_PATH, PROJECT_ROOT, VerneRNN, build_vocabulary

MODELS_DIR = PROJECT_ROOT / "models"

__all__ = ["ModelEntry", "ModelRegistry", "build_registry", "DEFAULT_MODELS_DIR"]

DEFAULT_MODELS_DIR = MODELS_DIR


class ModelAdapter(Protocol):
    """What the web app needs from any model, regardless of architecture."""

    key: str
    label: str
    architecture: str
    vocab_size: int
    parameter_count: int
    context_length: int
    model_file: str
    training_loss: float | None

    @property
    def char_to_idx(self) -> dict[str, int]: ...

    def generate(self, seed_text: str, num_generate: int = 800,
                 temperature: float = 0.7, top_k: int | None = None,
                 greedy: bool = False,
                 rng: np.random.Generator | None = None) -> Iterator[str]: ...


@dataclass
class ModelEntry:
    """One loadable model plus the metadata the UI needs."""

    key: str
    label: str
    architecture: str
    model: ModelAdapter
    path: Path
    blurb: str = ""

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count

    @property
    def context_length(self) -> int:
        return self.model.context_length

    @property
    def training_loss(self) -> float | None:
        return getattr(self.model, "training_loss", None)

    @property
    def model_file(self) -> str:
        return self.path.name


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

# Friendly names for the models this project ships or trains, so the UI reads
# well without the user having to name files carefully.
KNOWN_LABELS = {
    "verne_rnn_model": ("rnn", "RNN · GRU 1024", "Character GRU, 120-char window"),
    "smoke": ("smoke", "Transformer (smoke test)", "Tiny model from the notebook smoke test"),
    "tx-paired-ctx120": ("tx-paired-ctx120", "Transformer · matched, ctx 120",
                         "Size-matched to the RNN, same 120-char window"),
    "tx-paired": ("tx-paired", "Transformer · matched, ctx 256",
                  "Size-matched to the RNN, longer context"),
    "tx-large": ("tx-large", "Transformer · large", "25M parameters, for the scale story"),
}


def _display_name(stem: str, architecture: str) -> tuple[str, str]:
    if stem in KNOWN_LABELS:
        _, label, blurb = KNOWN_LABELS[stem]
        return label, blurb
    return f"{architecture.upper()} · {stem}", stem


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


class _RNNAdapter:
    """Uniform view over :class:`vernebot.engine.VerneRNN`."""

    architecture = "rnn"

    def __init__(self, model: VerneRNN) -> None:
        self._m = model

    @property
    def vocab_size(self) -> int:
        return self._m.vocab_size

    @property
    def parameter_count(self) -> int:
        return self._m.parameter_count

    @property
    def context_length(self) -> int:
        from .engine import CONTEXT_LENGTH

        return CONTEXT_LENGTH

    @property
    def training_loss(self) -> float | None:
        return None      # not stored in the .keras archive

    @property
    def char_to_idx(self) -> dict[str, int]:
        return self._m.char_to_idx

    def generate(self, seed_text: str, num_generate: int = 800,
                 temperature: float = 0.7, top_k: int | None = None,
                 greedy: bool = False,
                 rng: np.random.Generator | None = None) -> Iterator[str]:
        return self._m.generate(seed_text, num_generate=num_generate,
                                temperature=temperature, top_k=top_k,
                                greedy=greedy, rng=rng)


class _TransformerAdapter:
    """Uniform view over :class:`vernebot.transformer.VerneTransformer`."""

    architecture = "transformer"

    def __init__(self, model, vocab: list[str]) -> None:
        self._m = model
        self._vocab = vocab

    @property
    def vocab_size(self) -> int:
        return self._m.vocab_size

    @property
    def parameter_count(self) -> int:
        return self._m.parameter_count

    @property
    def context_length(self) -> int:
        return self._m.config.context

    @property
    def training_loss(self) -> float | None:
        return self._m.config.training_loss

    @property
    def char_to_idx(self) -> dict[str, int]:
        return self._m.char_to_idx

    def generate(self, seed_text: str, num_generate: int = 800,
                 temperature: float = 0.7, top_k: int | None = None,
                 greedy: bool = False,
                 rng: np.random.Generator | None = None) -> Iterator[str]:
        return self._m.generate(seed_text, num_generate=num_generate,
                                temperature=temperature, top_k=top_k,
                                greedy=greedy, rng=rng)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class ModelRegistry:
    """All loadable models, with lazy loading so a missing one cannot break startup."""

    def __init__(self, source: list[tuple[str, Path, str]] | None = None) -> None:
        self._source = source or []
        self._loaded: dict[str, ModelEntry] = {}
        self._failures: dict[str, str] = {}

    # -- discovery ---------------------------------------------------------

    @classmethod
    def discover(cls, models_dir: Path | str = MODELS_DIR,
                 vocab: list[str] | None = None) -> "ModelRegistry":
        """Find model files and build the registry (without loading weights yet)."""
        models_dir = Path(models_dir)
        source: list[tuple[str, Path, str]] = []

        if models_dir.is_dir():
            # The RNN first: it is the baseline everything is compared against.
            for path in sorted(models_dir.glob("*.keras")):
                if path.name.endswith(".keras.zip"):
                    continue
                source.append(("rnn", path, "keras"))
            for path in sorted(models_dir.glob("*.h5")):
                # Skip weights-only files that are not exported transformers.
                if "weights" in path.stem:
                    continue
                source.append(("transformer", path, "h5"))

        return cls(source)

    # -- loading -----------------------------------------------------------

    def _load(self, architecture: str, path: Path) -> ModelAdapter:
        if architecture == "rnn":
            return _RNNAdapter(VerneRNN.load(path))
        if architecture == "transformer":
            from .transformer import VerneTransformer

            return _TransformerAdapter(VerneTransformer.load(path), [])
        raise ValueError(f"unknown architecture {architecture!r}")

    def get(self, key: str) -> ModelEntry:
        """Return a loaded model, loading it on first use."""
        if key in self._loaded:
            return self._loaded[key]
        if key in self._failures:
            raise KeyError(f"{key}: {self._failures[key]}")

        for architecture, path, _kind in self._source:
            stem = path.stem
            if stem != key:
                continue
            try:
                model = self._load(architecture, path)
            except Exception as exc:      # noqa: BLE001 - reported to the caller
                self._failures[key] = f"{type(exc).__name__}: {exc}"
                raise KeyError(self._failures[key]) from exc

            label, blurb = _display_name(stem, architecture)
            entry = ModelEntry(key=key, label=label, architecture=architecture,
                               model=model, path=path, blurb=blurb)
            self._loaded[key] = entry
            return entry

        raise KeyError(f"unknown model {key!r}")

    # -- introspection -----------------------------------------------------

    def keys(self) -> list[str]:
        return [path.stem for _arch, path, _kind in self._source]

    def available(self) -> list[dict]:
        """Model list for the UI. Loads nothing, so it is safe at startup."""
        out = []
        for architecture, path, _kind in self._source:
            label, blurb = _display_name(path.stem, architecture)
            out.append({
                "key": path.stem,
                "label": label,
                "blurb": blurb,
                "architecture": architecture,
                "model_file": path.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                # Transformers are ~100x slower per character in NumPy, so the
                # per-request ceiling differs by architecture.
                "max_generate": 1500 if architecture == "transformer" else 4000,
            })
        return out

    def __len__(self) -> int:
        return len(self._source)

    def __contains__(self, key: object) -> bool:
        return key in self.keys()


def build_registry(models_dir: Path | str | None = None) -> ModelRegistry:
    """Discover models, honouring the ``VERNE_MODELS_DIR`` override."""
    directory = models_dir or os.environ.get("VERNE_MODELS_DIR") or MODELS_DIR
    return ModelRegistry.discover(directory)
