#!/usr/bin/env python3
"""
Train the Jules Verne Transformer.

The logic lives here, in a reusable module with a command-line interface, and
the Colab notebook is a thin wrapper around it. That means the exact same code
path produces the model whether it is run in a notebook cell or on a GPU box,
and the command used for any given checkpoint can be reproduced from the
arguments alone.

    python train_transformer.py --preset paired --steps 4000 --out models/tx-paired.h5
    python train_transformer.py --preset paired --context 120 --steps 4000
    python train_transformer.py --preset large  --steps 12000 --batch-size 48

Fair-comparison rules this script is built around
-----------------------------------------------
* The tokenizer is identical to the RNN's: characters, vocabulary from
  ``books/``, joined exactly as the RNN notebook joined them. Only the
  architecture varies.
* ``--context`` defaults to the config's value (256). Pass ``--context 120`` for
  the purest architecture comparison, since 120 is the RNN's window.
* The trained tokenizer, the architecture config and the parameter count are all
  written into the exported file, so the website can load a model without being
  told anything about it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vernebot.engine import BOOKS_DIR, build_vocabulary
from vernebot.transformer_model import (
    LARGE,
    PAIRED,
    PAIRED_CONTEXT_120,
    TransformerConfig,
    VerneTransformer,
    export_weights,
)

PROJECT_ROOT = Path(__file__).resolve().parent

PRESETS: dict[str, TransformerConfig] = {
    "paired": PAIRED,
    "paired-ctx120": PAIRED_CONTEXT_120,
    "large": LARGE,
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_corpus(books_dir: Path | str = BOOKS_DIR) -> str:
    """Concatenate the books exactly as the RNN notebook did.

    The RNN notebook read every ``*.txt`` in ``books/`` and appended a single
    newline after each. We keep that, and additionally separate books with a
    blank line so the model can learn a document boundary.
    """
    books_dir = Path(books_dir)
    parts = []
    for filename in sorted(os.listdir(books_dir)):
        if filename.endswith(".txt"):
            with open(books_dir / filename, "r", encoding="utf-8") as fh:
                parts.append(fh.read().rstrip("\n"))
    return "\n\n".join(parts) + "\n"


def encode(text: str, vocab: list[str]) -> np.ndarray:
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    missing = sorted(set(text) - set(char_to_idx))
    if missing:
        # Should never happen when the corpus is books/ itself, but a clear
        # error beats silently producing a wrong vocabulary mapping.
        raise ValueError(f"Corpus contains characters outside the vocabulary: {missing!r}")
    return np.array([char_to_idx[c] for c in text], dtype=np.uint16)


def split_corpus(data: np.ndarray, val_fraction: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    cut = int(len(data) * (1.0 - val_fraction))
    return data[:cut], data[cut:]


def get_batch(data: np.ndarray, context: int, batch_size: int,
              rng: np.random.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous windows.

    ``x`` is the window and ``y`` is the same window shifted one character,
    which is the character-level next-token objective.
    """
    starts = rng.integers(0, len(data) - context - 1, size=batch_size)
    x = np.stack([data[s: s + context] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1: s + 1 + context] for s in starts]).astype(np.int64)
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


# --------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay, as in GPT-style training recipes."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak * (min_ratio + (1.0 - min_ratio) * coeff)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(model: VerneTransformer, train: np.ndarray, val: np.ndarray,
                  context: int, batch_size: int, eval_batches: int,
                  rng: np.random.Generator, device: torch.device) -> dict[str, float]:
    model.eval()
    out = {}
    for label, data in (("train", train), ("val", val)):
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, context, batch_size, rng, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        out[label] = float(np.mean(losses))
    model.train()
    return out


@torch.no_grad()
def sample_text(model: VerneTransformer, vocab: list[str], seed: str,
                n: int, temperature: float, device: torch.device) -> str:
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    tokens = [char_to_idx[c] for c in seed if c in char_to_idx]
    idx = torch.tensor([tokens], device=device)
    out = model.generate(idx, max_new_tokens=n, temperature=temperature)
    return "".join(vocab[int(t)] for t in out[0])


def train(
    cfg: TransformerConfig,
    *,
    steps: int = 4000,
    batch_size: int = 64,
    lr: float = 3e-3,
    warmup: int = 200,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    eval_every: int = 250,
    eval_batches: int = 40,
    val_fraction: float = 0.05,
    sample_every: int = 1000,
    sample_seed: str = "THE FLYING SUBMARINE",
    sample_length: int = 300,
    sample_temperature: float = 0.7,
    out_path: Path | str | None = None,
    books_dir: Path | str = BOOKS_DIR,
    device: str | None = None,
    seed: int = 1337,
    log=print,
) -> dict:
    """Train one model and export it. Returns a summary dict."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    vocab = build_vocabulary(books_dir)
    if len(vocab) != cfg.vocab_size:
        raise ValueError(
            f"Vocabulary has {len(vocab)} characters but the config expects "
            f"{cfg.vocab_size}. Refusing to train on a mismatched vocabulary."
        )

    text = load_corpus(books_dir)
    data = encode(text, vocab)
    train_data, val_data = split_corpus(data, val_fraction)

    log(f"device        : {device}")
    log(f"corpus        : {len(text):,} characters")
    log(f"vocab         : {len(vocab)} characters")
    log(f"train / val   : {len(train_data):,} / {len(val_data):,}")
    log(f"context       : {cfg.context}")
    log(f"batch x steps : {batch_size} x {steps}")

    model = VerneTransformer(cfg).to(dev)
    n_params = model.parameter_count()
    log(f"parameters    : {n_params:,}   (RNN baseline: 4,095,867 -- ratio "
        f"{n_params / 4095867:.2f}x)")

    # AdamW with weight decay only on matrices, the usual convention.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)
    optimiser = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )

    history: list[dict] = []
    samples: list[dict] = []
    started = time.time()

    for step in range(steps):
        for group in optimiser.param_groups:
            group["lr"] = lr_at(step, steps, lr, warmup)

        x, y = get_batch(train_data, cfg.context, batch_size, rng, dev)
        _, loss = model(x, y)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()

        if step % eval_every == 0 or step == steps - 1:
            metrics = estimate_loss(model, train_data, val_data, cfg.context,
                                    batch_size, eval_batches, rng, dev)
            entry = {"step": step, "lr": optimiser.param_groups[0]["lr"], **metrics,
                     "elapsed_s": round(time.time() - started, 1)}
            history.append(entry)
            log(f"step {step:>6d}/{steps}  train {metrics['train']:.4f}  "
                f"val {metrics['val']:.4f}  ppl {math.exp(metrics['val']):8.2f}  "
                f"lr {entry['lr']:.2e}  [{entry['elapsed_s']}s]")

        if sample_every and (step + 1) % sample_every == 0:
            text_sample = sample_text(model, vocab, sample_seed, sample_length,
                                      sample_temperature, dev)
            samples.append({"step": step + 1, "text": text_sample})
            log(f"\n  --- sample at step {step + 1} ---")
            log("  " + text_sample[:400].replace("\n", "\n  ") + "\n")

    final = estimate_loss(model, train_data, val_data, cfg.context,
                          batch_size, max(eval_batches, 100), rng, dev)
    log(f"\nfinal: train {final['train']:.4f}  val {final['val']:.4f}  "
        f"perplexity {math.exp(final['val']):.2f}")

    summary = {
        "config": asdict(cfg),
        "parameters": n_params,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "device": device,
        "final_train_loss": final["train"],
        "final_val_loss": final["val"],
        "final_val_perplexity": math.exp(final["val"]),
        "train_minutes": round((time.time() - started) / 60, 2),
        "torch_version": torch.__version__,
        "history": history,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_weights(
            model, out_path, vocab=vocab,
            extra_metadata={
                "training_loss": final["val"],
                "training": {k: v for k, v in summary.items() if k != "history"},
                "corpus_characters": len(text),
                "sample_seed": sample_seed,
            },
        )
        log(f"\nexported -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

        # Persist the run record next to the weights so experiments stay comparable.
        record = out_path.with_suffix(".json")
        record.write_text(json.dumps({**summary, "samples": samples}, indent=2))
        log(f"run record -> {record}")

    summary["samples"] = samples
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Jules Verne Transformer.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="paired",
                   help="paired = size-matched to the RNN; large = scale demo.")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--context", type=int, default=None,
                   help="Override context length (use 120 to match the RNN exactly).")
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--positional", choices=["learned", "sinusoidal"], default=None)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto by default).")
    p.add_argument("--books-dir", default=str(BOOKS_DIR))
    p.add_argument("--out", default=None, help="Output .h5 path.")
    p.add_argument("--name", default=None, help="Model name stored in the file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PRESETS[args.preset]

    overrides = {
        "context": args.context,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "positional": args.positional,
        "name": args.name,
    }
    cfg = TransformerConfig(**{
        **asdict(cfg),
        **{k: v for k, v in overrides.items() if v is not None},
    })

    out = args.out or (PROJECT_ROOT / "models" / f"{cfg.name}.h5")
    train(
        cfg,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        sample_every=args.sample_every,
        seed=args.seed,
        device=args.device,
        books_dir=args.books_dir,
        out_path=out,
    )


if __name__ == "__main__":
    main()
