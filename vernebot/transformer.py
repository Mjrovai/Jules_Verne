"""
Pure NumPy inference for the trained Jules Verne Transformer.

This is the Transformer counterpart of :mod:`vernebot.engine`: it reads the
framework-neutral HDF5 file written by ``vernebot/transformer_model.py`` and
runs a decoder-only Transformer with **no PyTorch and no TensorFlow** at
serving time, so the website's dependency list does not change.

Two things are worth reading closely, because they are the Transformer's answer
to the RNN's hidden-state trick.

**KV cache.** Without it, generating token *n* reprocesses all *n* earlier
tokens, so cost grows with the length of the text and generation visibly slows
down. With it, each step computes keys and values for one token and appends
them -- the same "carry the past forward" idea as the GRU hidden state.

**The window slides, and that constrains what may be cached.** Keys and values
for a past token were computed *in the context of everything before it*, so once
generation passes ``context`` and the window moves, cached entries no longer
describe what a fresh forward pass would compute. Reusing them naively drifts:
measured on a test model it agreed with a sliding-window reference for only 55%
of tokens, diverging exactly at the context boundary.

**Ring buffer, not re-warming.** The fix is *not* to replay the window for every
new character. That is exact but pathological: each character costs ``context``
passes through the stack, so at C=120 the cost per character jumped from ~3 ms to
**340 ms** as soon as the window filled (measured: 367 characters took 122 s by
replay versus 1.2 s with the ring).

Instead the cache is a ring. A new character overwrites the oldest slot,
attention skips that slot (after the overwrite it holds a *newer* token, which
must not be attended to), and each character costs exactly one pass. Cost per
character is then flat -- measured at ~3.5 ms for the 4M models and ~29 ms for
the 25M one, independent of how long the text has become.

Two details make it correct rather than merely fast:

* ``slot = pos % context`` doubles as the token's **position in the current
  window**. Once the ring wraps, positions restart at 0, so the window is always
  "the last C characters, numbered from its own start" -- which is what training
  saw. Using the absolute position here instead assigns a generated token the
  position embedding of a much later slot and the output collapses to gibberish.
* The prompt is limited to ``context - 1`` characters, reserving one slot, so the
  first generated token never lands on a position a prompt token already used.

**The approximation this buys.** Because the loop feeds one token at a time, the
cache keeps the prompt tokens' key/values as computed at their original
positions. A full replay would recompute them at renumbered positions as the
window slides, so the two agree exactly while the window is filling and diverge
at the first wrap (verified: identical for the first 119 generated characters,
then differing). The ring path stays coherent -- it is the standard sliding-window
scheme -- but it is an approximation, not bit-identical to replay. That trade is
deliberate: replay is 100x slower and unusable interactively.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np

__all__ = ["TransformerConfig", "TransformerWeights", "LayerWeights", "VerneTransformer"]


# --------------------------------------------------------------------------
# Configuration and weights
# --------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    """Architecture read from the exported file's ``config`` attribute."""

    name: str = "transformer"
    vocab_size: int = 123
    context: int = 256
    d_model: int = 256
    n_layers: int = 5
    n_heads: int = 4
    d_ff: int = 1024
    positional: str = "learned"
    tie_embeddings: bool = True
    parameter_count: int = 0
    framework: str = "pytorch"
    training_loss: float | None = None

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class LayerWeights:
    """All tensors for one pre-norm transformer block.

    Linear weights are stored as PyTorch does, ``(out_features, in_features)``,
    so the forward pass uses ``x @ W.T``.
    """

    ln1_g: np.ndarray
    ln1_b: np.ndarray
    wq_w: np.ndarray
    wq_b: np.ndarray
    wk_w: np.ndarray
    wk_b: np.ndarray
    wv_w: np.ndarray
    wv_b: np.ndarray
    wo_w: np.ndarray
    wo_b: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    fc_w: np.ndarray
    fc_b: np.ndarray
    proj_w: np.ndarray
    proj_b: np.ndarray


@dataclass
class TransformerWeights:
    wte: np.ndarray
    wpe: np.ndarray | None
    lnf_g: np.ndarray
    lnf_b: np.ndarray
    head: np.ndarray
    layers: list[LayerWeights] = field(default_factory=list)


# --------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------


def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact (erf-based) GELU, matching ``torch.nn.functional.gelu``'s default.

    This is what PyTorch trained with, and serving uses it too. An earlier
    version of this module substituted the tanh approximation because it is
    about 5x faster on a 1024-element vector -- but the whole model got only 5%
    faster, while the approximation is systematically biased rather than random,
    so five layers of residual connections compounded it: measured against
    PyTorch, the tanh form drifts to 0.048% by a 20-character prompt while this
    one stays at 0.0000%.

    That mattered most for the browser port, where the same approximation had
    been copied independently and produced visibly worse text. Being faithful to
    the trained model is worth 5%.
    """
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU, tanh approximation. Kept only so the difference can be measured."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _sinusoidal_positions(context: int, d_model: int) -> np.ndarray:
    """Reproduce the training-side sinusoidal table for non-learned positions."""
    idx = np.arange(context, dtype=np.float64)[:, None]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * (-math.log(10000.0) / d_model))
    pe = np.zeros((context, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(idx * div)
    pe[:, 1::2] = np.cos(idx * div)
    return pe


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class VerneTransformer:
    """Decoder-only Transformer loaded from an exported HDF5 file."""

    def __init__(self, weights: TransformerWeights, config: TransformerConfig,
                 vocab: list[str], model_path: Path | None = None,
                 exact_gelu: bool = False) -> None:
        self.w = weights
        self.config = config
        self.vocab = vocab
        self.model_path = model_path
        self._char_to_idx = {c: i for i, c in enumerate(vocab)}
        self._pos = (weights.wpe if weights.wpe is not None
                     else _sinusoidal_positions(config.context, config.d_model))
        # Serving uses the fast tanh GELU; parity tests can ask for the exact one.
        self._gelu_fn = _gelu_tanh if exact_gelu else _gelu

    # -- properties --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def char_to_idx(self) -> dict[str, int]:
        return self._char_to_idx

    @property
    def context(self) -> int:
        return self.config.context

    @property
    def parameter_count(self) -> int:
        if self.config.parameter_count:
            return self.config.parameter_count
        total = self.w.wte.size + self.w.head.size + self.w.lnf_g.size + self.w.lnf_b.size
        if self.w.wpe is not None:
            total += self.w.wpe.size
        for layer in self.w.layers:
            total += sum(
                getattr(layer, f).size
                for f in ("ln1_g", "ln1_b", "wq_w", "wq_b", "wk_w", "wk_b", "wv_w", "wv_b",
                          "wo_w", "wo_b", "ln2_g", "ln2_b", "fc_w", "fc_b", "proj_w", "proj_b")
            )
        return total

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, model_path: str | Path,
             vocab: Sequence[str] | None = None) -> "VerneTransformer":
        """Read an exported transformer from HDF5."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Transformer weights not found: {model_path}")

        with h5py.File(model_path, "r") as f:
            raw_config = f.attrs.get("config")
            if raw_config is None:
                raise ValueError(
                    f"{model_path.name} has no 'config' attribute; it does not look like a "
                    "file exported by vernebot.transformer_model.export_weights()."
                )
            meta = json.loads(raw_config)
            known = TransformerConfig.__dataclass_fields__
            cfg = TransformerConfig(**{k: v for k, v in meta.get("config", {}).items()
                                       if k in known})
            cfg.parameter_count = int(meta.get("parameter_count", 0))
            cfg.framework = meta.get("framework", "unknown")
            cfg.training_loss = meta.get("training_loss")

            if vocab is None:
                stored = meta.get("vocab")
                if stored is None:
                    raise ValueError(
                        "No vocabulary stored in the file and none supplied. Pass "
                        "vocab=... or re-export with vocab=..."
                    )
                vocab = list(stored)

            def arr(name: str) -> np.ndarray:
                return np.asarray(f[name], dtype=np.float32)

            wte = arr("embedding/wte")
            wpe = arr("embedding/wpe") if "embedding/wpe" in f else None
            lnf_g = arr("final_norm/weight")
            lnf_b = arr("final_norm/bias")
            head = arr("head/weight")

            layers: list[LayerWeights] = []
            for i in range(cfg.n_layers):
                p = f"blocks/{i}"
                layers.append(LayerWeights(
                    ln1_g=arr(f"{p}/ln1/weight"), ln1_b=arr(f"{p}/ln1/bias"),
                    wq_w=arr(f"{p}/attn/wq/weight"), wq_b=arr(f"{p}/attn/wq/bias"),
                    wk_w=arr(f"{p}/attn/wk/weight"), wk_b=arr(f"{p}/attn/wk/bias"),
                    wv_w=arr(f"{p}/attn/wv/weight"), wv_b=arr(f"{p}/attn/wv/bias"),
                    wo_w=arr(f"{p}/attn/wo/weight"), wo_b=arr(f"{p}/attn/wo/bias"),
                    ln2_g=arr(f"{p}/ln2/weight"), ln2_b=arr(f"{p}/ln2/bias"),
                    fc_w=arr(f"{p}/mlp/fc/weight"), fc_b=arr(f"{p}/mlp/fc/bias"),
                    proj_w=arr(f"{p}/mlp/proj/weight"), proj_b=arr(f"{p}/mlp/proj/bias"),
                ))

        if wte.shape != (len(vocab), cfg.d_model):
            raise ValueError(
                f"Vocabulary/model mismatch: the model expects {wte.shape[0]} tokens but "
                f"{len(vocab)} were supplied."
            )

        return cls(TransformerWeights(wte, wpe, lnf_g, lnf_b, head, layers),
                   cfg, list(vocab), model_path)

    # -- forward pass ------------------------------------------------------

    def _fresh_cache(self) -> list[dict[str, np.ndarray]]:
        cfg = self.config
        return [{"k": np.zeros((cfg.context, cfg.d_model), np.float32),
                 "v": np.zeros((cfg.context, cfg.d_model), np.float32)}
                for _ in range(cfg.n_layers)]

    def forward(self, tokens: Sequence[int],
                cache: list[dict[str, np.ndarray]] | None = None,
                start: int = 0) -> tuple[np.ndarray, list, int]:
        """Run ``tokens`` through the stack, writing their keys/values to ``cache``.

        ``start`` is the absolute position of the first token; it must equal the
        number of tokens already in the cache. Every token attends to all earlier
        ones, exactly as during training, so this is how a prompt is ingested.
        Returns ``(logits_at_last_token, cache, next_start)``.

        The cache may not exceed the context length. To generate beyond it,
        continue with :meth:`forward_step`, which overwrites the oldest entries
        in a ring instead of replaying the window.
        """
        cfg = self.config
        if cache is None:
            cache = self._fresh_cache()
            start = 0
        if start + len(tokens) > cfg.context:
            raise ValueError(
                f"{start} tokens cached + {len(tokens)} new would exceed the context "
                f"length {cfg.context}; use forward_step() to generate beyond it."
            )

        logits = np.zeros(cfg.vocab_size, dtype=np.float32)
        for step, token in enumerate(tokens):
            pos = start + step
            x = self.w.wte[token] + self._pos[pos]

            for li, layer in enumerate(self.w.layers):
                normed = _layer_norm(x, layer.ln1_g, layer.ln1_b)
                q = normed @ layer.wq_w.T + layer.wq_b
                k = normed @ layer.wk_w.T + layer.wk_b
                v = normed @ layer.wv_w.T + layer.wv_b
                cache[li]["k"][pos] = k
                cache[li]["v"][pos] = v

                K = cache[li]["k"][: pos + 1].reshape(pos + 1, cfg.n_heads, cfg.d_head)
                V = cache[li]["v"][: pos + 1].reshape(pos + 1, cfg.n_heads, cfg.d_head)
                qh = q.reshape(cfg.n_heads, cfg.d_head)
                scores = np.einsum("hd,thd->ht", qh, K) * (1.0 / math.sqrt(cfg.d_head))
                weights = _softmax(scores)
                head_out = np.einsum("ht,thd->hd", weights, V).reshape(cfg.d_model)

                x = x + head_out @ layer.wo_w.T + layer.wo_b

                normed = _layer_norm(x, layer.ln2_g, layer.ln2_b)
                hidden = self._gelu_fn(normed @ layer.fc_w.T + layer.fc_b)
                x = x + hidden @ layer.proj_w.T + layer.proj_b

            logits = self.w.head @ _layer_norm(x, self.w.lnf_g, self.w.lnf_b)

        return logits, cache, start + len(tokens)

    def forward_step(self, token: int, cache: list[dict[str, np.ndarray]],
                     pos: int) -> np.ndarray:
        """One generated character: the ring path.

        ``pos`` is the absolute position and keeps counting past the context
        length. The new key/value overwrites slot ``pos % context`` -- the oldest
        entry -- and attention skips that slot, because after the overwrite it
        holds a *newer* token that must not be visible to this one. The slot, not
        ``pos``, indexes the learned position table, which is what renumbers the
        window from its start.

        Cost is one pass through the stack, independent of how long the text has
        become. Replaying the window instead would cost ``context`` passes per
        character -- measured at 173 ms/character versus 3 ms flat.
        """
        cfg = self.config

        # The slot doubles as the token's position in the current window. Once
        # the ring wraps, position numbers restart at 0, so the window is always
        # "the last `context` characters, numbered from its own start" -- which
        # is what the model was trained on (fixed windows, positions 0..C-1).
        #
        # If the absolute position were used instead, a generated token would be
        # assigned the position embedding of a much later slot and the output
        # would collapse into gibberish.
        slot = pos % cfg.context
        x = self.w.wte[token] + self._pos[slot]

        if pos >= cfg.context:
            # Wrapped: every slot holds a live token, and the one we are about
            # to overwrite now holds *this* token, so it must be excluded.
            idx = np.arange(cfg.context)
            idx = idx[idx != slot]
        else:
            idx = None

        for li, layer in enumerate(self.w.layers):
            normed = _layer_norm(x, layer.ln1_g, layer.ln1_b)
            q = normed @ layer.wq_w.T + layer.wq_b
            k = normed @ layer.wk_w.T + layer.wk_b
            v = normed @ layer.wv_w.T + layer.wv_b
            cache[li]["k"][slot] = k
            cache[li]["v"][slot] = v

            if idx is None:
                rows = pos + 1
                K = cache[li]["k"][:rows].reshape(rows, cfg.n_heads, cfg.d_head)
                V = cache[li]["v"][:rows].reshape(rows, cfg.n_heads, cfg.d_head)
            else:
                # Gather first, then shape from the gathered length so the two
                # can never disagree.
                K = cache[li]["k"][idx]
                V = cache[li]["v"][idx]
                rows = K.shape[0]
                K = K.reshape(rows, cfg.n_heads, cfg.d_head)
                V = V.reshape(rows, cfg.n_heads, cfg.d_head)

            qh = q.reshape(cfg.n_heads, cfg.d_head)
            scores = np.einsum("hd,thd->ht", qh, K) * (1.0 / math.sqrt(cfg.d_head))
            weights = _softmax(scores)
            head_out = np.einsum("ht,thd->hd", weights, V).reshape(cfg.d_model)

            x = x + head_out @ layer.wo_w.T + layer.wo_b

            normed = _layer_norm(x, layer.ln2_g, layer.ln2_b)
            hidden = self._gelu_fn(normed @ layer.fc_w.T + layer.fc_b)
            x = x + hidden @ layer.proj_w.T + layer.proj_b

        return self.w.head @ _layer_norm(x, self.w.lnf_g, self.w.lnf_b)

    def logits(self, tokens: Sequence[int], tanh_gelu: bool = False) -> np.ndarray:
        """Convenience: logits for a prompt, with no cache returned."""
        if tanh_gelu:
            self._gelu_fn = _gelu_tanh
        logits, _, _ = self.forward(tokens)
        return logits

    # -- generation --------------------------------------------------------

    def generate(self, seed_text: str, num_generate: int = 800,
                 temperature: float = 0.7, top_k: int | None = None,
                 greedy: bool = False,
                 rng: np.random.Generator | None = None) -> Iterator[str]:
        """Yield the seed text, then one generated character at a time.

        The prompt fills the cache with ``ring=False`` (each prompt token
        attends to every earlier one, matching training). Generation then
        continues in ring mode: each new character overwrites the oldest cache
        slot and costs exactly one pass through the stack, so the cost per
        character does not grow with how long the text has become.
        """
        cfg = self.config
        cleaned = "".join(c for c in seed_text if c in self._char_to_idx)
        if not cleaned:
            raise ValueError("Seed text must contain at least one in-vocabulary character.")

        rng = rng or np.random.default_rng()

        # Reserve one slot: the prompt may occupy at most context-1 positions so
        # that the first generated token lands on a fresh one. Without this, a
        # prompt of exactly `context` characters would put the first generated
        # token at position 0 -- the same position a prompt token already used,
        # which the model reads as a token that does not exist.
        prompt = [self._char_to_idx[c] for c in cleaned[-(cfg.context - 1):]]
        logits, cache, start = self.forward(prompt)

        yield cleaned

        # Generation continues from where the prompt ended; positions keep
        # counting up and the slot (`pos % context`) is what indexes the learned
        # position table, so the window is renumbered once it wraps.
        pos = start
        for _ in range(int(num_generate)):
            index = self._sample(logits, temperature, top_k, greedy, rng)
            yield self.vocab[index]
            logits = self.forward_step(index, cache, pos)
            pos += 1

    def _sample(self, logits: np.ndarray, temperature: float,
                top_k: int | None, greedy: bool,
                rng: np.random.Generator) -> int:
        if greedy:
            return int(np.argmax(logits))

        logits = logits.astype(np.float64) / max(float(temperature), 1e-4)
        if top_k is not None and 0 < top_k < logits.shape[0]:
            keep = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[keep] = logits[keep]
            logits = mask

        logits -= logits.max()
        probs = np.exp(logits)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return int(np.argmax(logits))
        probs /= total
        return int(rng.choice(probs.shape[0], p=probs))


def load_transformer(model_path: str | Path,
                     vocab: Sequence[str] | None = None) -> VerneTransformer:
    """Convenience wrapper around :meth:`VerneTransformer.load`."""
    return VerneTransformer.load(model_path, vocab)
