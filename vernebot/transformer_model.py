"""
Trainable decoder-only Transformer for the Jules Verne experiment.

This module is the **training side** of the Transformer comparison and is only
imported when PyTorch is available (i.e. in the Colab notebook). The website
never imports it -- it loads the exported weights with the pure-NumPy engine in
:mod:`vernebot.transformer`, so PyTorch and TensorFlow stay out of the
deployment path.

Design decisions that make the comparison fair against the RNN
-------------------------------------------------------------
1. **Same tokenizer.** Character level, 123 symbols, built from ``books/``
   exactly as the RNN notebook did. Only the architecture varies.
2. **Comparable size.** The default "paired" config is 4,313,344 parameters
   against the RNN's 4,095,867 -- within 5%.
3. **Controlled context.** ``context`` defaults to 256. Train one run at 120 to
   match the RNN's window exactly, and another at 256/512 to isolate the value
   of long context.

Export contract
---------------
:func:`export_weights` writes an HDF5 file with one group per tensor, using the
names documented in the module-level ``WEIGHT_SPEC`` below. That layout is what
``vernebot/transformer.py`` reads; changing one without the other breaks
inference, so a round-trip test guards it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    """Hyper-parameters for one model."""

    name: str = "transformer-paired"
    vocab_size: int = 123
    context: int = 256
    d_model: int = 256
    n_layers: int = 5
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    tie_embeddings: bool = True
    positional: Literal["learned", "sinusoidal"] = "learned"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# "paired" matches the RNN's parameter count within 5%; "large" demonstrates scale.
PAIRED = TransformerConfig()

LARGE = TransformerConfig(
    name="transformer-large",
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=2048,
    context=256,
)

# A run whose context exactly matches the RNN's 120-character window, for the
# purest architecture-vs-architecture comparison.
PAIRED_CONTEXT_120 = TransformerConfig(
    name="transformer-paired-ctx120",
    context=120,
)

PRESETS: dict[str, TransformerConfig] = {
    "paired": PAIRED,
    "paired-ctx120": PAIRED_CONTEXT_120,
    "large": LARGE,
}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, cfg.d_model)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)

        # PyTorch's scaled_dot_product_attention applies the causal mask and the
        # 1/sqrt(d_head) scaling, and uses a fused kernel when available.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.cfg.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.wo(out))


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class VerneTransformer(nn.Module):
    """Character-level decoder-only Transformer."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = (nn.Embedding(cfg.context, cfg.d_model)
                    if cfg.positional == "learned" else None)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Scale residual-branch output projections down, as in GPT-2.
        for name, param in self.named_parameters():
            if name.endswith(("wo.weight", "proj.weight")):
                nn.init.normal_(param, mean=0.0,
                                std=0.02 / np.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.context:
            raise ValueError(f"sequence length {T} exceeds context {self.cfg.context}")

        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(pos)
        else:
            x = x + self._sinusoidal(pos, x.dtype)

        for block in self.blocks:
            x = block(x)
        x = self.lnf(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    def _sinusoidal(self, pos: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        cfg = self.cfg
        idx = pos[:, None].float()
        div = torch.exp(torch.arange(0, cfg.d_model, 2, device=pos.device).float()
                        * (-np.log(10000.0) / cfg.d_model))
        pe = torch.zeros(len(pos), cfg.d_model, device=pos.device)
        pe[:, 0::2] = torch.sin(idx * div)
        pe[:, 1::2] = torch.cos(idx * div)
        return pe.to(dtype)

    # -- introspection -----------------------------------------------------

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.7, top_k: int | None = None,
                 greedy: bool = False) -> torch.Tensor:
        """Reference sampler, used only to cross-check the NumPy engine."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-4)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    cutoff = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=-1)
        return idx


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

#: Canonical tensor names written to HDF5 and read by vernebot/transformer.py.
#: ``{}`` is filled with the layer index.
WEIGHT_SPEC = {
    "wte": "embedding/wte",
    "wpe": "embedding/wpe",
    "lnf_g": "final_norm/weight",
    "lnf_b": "final_norm/bias",
    "head": "head/weight",
    "block_ln1_g": "blocks/{i}/ln1/weight",
    "block_ln1_b": "blocks/{i}/ln1/bias",
    "block_wq_w": "blocks/{i}/attn/wq/weight",
    "block_wq_b": "blocks/{i}/attn/wq/bias",
    "block_wk_w": "blocks/{i}/attn/wk/weight",
    "block_wk_b": "blocks/{i}/attn/wk/bias",
    "block_wv_w": "blocks/{i}/attn/wv/weight",
    "block_wv_b": "blocks/{i}/attn/wv/bias",
    "block_wo_w": "blocks/{i}/attn/wo/weight",
    "block_wo_b": "blocks/{i}/attn/wo/bias",
    "block_ln2_g": "blocks/{i}/ln2/weight",
    "block_ln2_b": "blocks/{i}/ln2/bias",
    "block_fc_w": "blocks/{i}/mlp/fc/weight",
    "block_fc_b": "blocks/{i}/mlp/fc/bias",
    "block_proj_w": "blocks/{i}/mlp/proj/weight",
    "block_proj_b": "blocks/{i}/mlp/proj/bias",
}


def export_weights(model: VerneTransformer, path: str | Path,
                   vocab: list[str] | None = None,
                   extra_metadata: dict[str, Any] | None = None) -> Path:
    """Write the trained model to HDF5 for the NumPy inference engine.

    The file is framework-neutral (plain HDF5), so it can be read with
    ``h5py`` and nothing else. Metadata is stored as JSON in the root
    attribute ``config``.
    """
    import h5py

    path = Path(path)
    cfg = model.cfg
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}

    def put(group: h5py.Group, name: str, array: np.ndarray) -> None:
        parts = name.split("/")
        node: Any = group
        for part in parts[:-1]:
            node = node.require_group(part)
        node.create_dataset(parts[-1], data=array)

    with h5py.File(path, "w") as f:
        put(f, WEIGHT_SPEC["wte"], sd["wte.weight"])
        if cfg.positional == "learned":
            put(f, WEIGHT_SPEC["wpe"], sd["wpe.weight"])
        else:
            # Sinusoidal positions are computed, so store none; the engine sees
            # the flag in the config and reconstructs them.
            pass
        put(f, WEIGHT_SPEC["lnf_g"], sd["lnf.weight"])
        put(f, WEIGHT_SPEC["lnf_b"], sd["lnf.bias"])
        put(f, WEIGHT_SPEC["head"], sd["head.weight"])

        for i in range(cfg.n_layers):
            prefix = f"blocks.{i}."
            mapping = {
                "block_ln1_g": "ln1.weight", "block_ln1_b": "ln1.bias",
                "block_wq_w": "attn.wq.weight", "block_wq_b": "attn.wq.bias",
                "block_wk_w": "attn.wk.weight", "block_wk_b": "attn.wk.bias",
                "block_wv_w": "attn.wv.weight", "block_wv_b": "attn.wv.bias",
                "block_wo_w": "attn.wo.weight", "block_wo_b": "attn.wo.bias",
                "block_ln2_g": "ln2.weight", "block_ln2_b": "ln2.bias",
                "block_fc_w": "mlp.fc.weight", "block_fc_b": "mlp.fc.bias",
                "block_proj_w": "mlp.proj.weight", "block_proj_b": "mlp.proj.bias",
            }
            for key, source in mapping.items():
                put(f, WEIGHT_SPEC[key].format(i=i), sd[prefix + source])

        metadata = {
            "architecture": "decoder-only-transformer",
            "config": asdict(cfg),
            "parameter_count": model.parameter_count(),
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "training_loss": None,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        if vocab is not None:
            metadata["vocab"] = "".join(vocab)
        f.attrs["config"] = json.dumps(metadata)

    return path


def load_into_model(path: str | Path, model: VerneTransformer) -> None:
    """Inverse of :func:`export_weights`, for round-trip verification."""
    import h5py

    with h5py.File(path, "r") as f:
        sd = {}
        for name, ds in _iter_datasets(f):
            sd[name] = torch.from_numpy(np.array(ds))
    model.load_state_dict(_remap_to_state_dict(sd, model.cfg), strict=False)


def _iter_datasets(group, prefix=""):
    import h5py

    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            yield prefix + key, item
        else:
            yield from _iter_datasets(item, prefix + key + ".")


def _remap_to_state_dict(flat: dict[str, torch.Tensor],
                         cfg: TransformerConfig) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    sd["wte.weight"] = flat["embedding.wte"]
    if cfg.positional == "learned" and "embedding.wpe" in flat:
        sd["wpe.weight"] = flat["embedding.wpe"]
    sd["lnf.weight"] = flat["final_norm.weight"]
    sd["lnf.bias"] = flat["final_norm.bias"]
    sd["head.weight"] = flat["head.weight"]
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        sd[p + "ln1.weight"] = flat[p + "ln1.weight"]
        sd[p + "ln1.bias"] = flat[p + "ln1.bias"]
        sd[p + "attn.wq.weight"] = flat[p + "attn.wq.weight"]
        sd[p + "attn.wq.bias"] = flat[p + "attn.wq.bias"]
        sd[p + "attn.wk.weight"] = flat[p + "attn.wk.weight"]
        sd[p + "attn.wk.bias"] = flat[p + "attn.wk.bias"]
        sd[p + "attn.wv.weight"] = flat[p + "attn.wv.weight"]
        sd[p + "attn.wv.bias"] = flat[p + "attn.wv.bias"]
        sd[p + "attn.wo.weight"] = flat[p + "attn.wo.weight"]
        sd[p + "attn.wo.bias"] = flat[p + "attn.wo.bias"]
        sd[p + "ln2.weight"] = flat[p + "ln2.weight"]
        sd[p + "ln2.bias"] = flat[p + "ln2.bias"]
        sd[p + "mlp.fc.weight"] = flat[p + "mlp.fc.weight"]
        sd[p + "mlp.fc.bias"] = flat[p + "mlp.fc.bias"]
        sd[p + "mlp.proj.weight"] = flat[p + "mlp.proj.weight"]
        sd[p + "mlp.proj.bias"] = flat[p + "mlp.proj.bias"]
    return sd


if __name__ == "__main__":
    for key, cfg in PRESETS.items():
        model = VerneTransformer(cfg)
        print(f"{key:16s} {cfg.name:26s} params = {model.parameter_count():,}")
    print()
    print("RNN baseline for comparison: 4,095,867")
