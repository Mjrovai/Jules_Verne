"""
Export a trained Jules Verne Transformer to ONNX for fast CPU/Metal inference.

STATUS: experimental, not part of the supported serving path.
------------------------------------------------------------
The pure-NumPy engine in :mod:`vernebot.transformer` is the supported path: it
is verified numerically against PyTorch (logits agree to ~1e-7, and greedy
generation matches token for token), needs no framework at serving time, and is
fast enough for the size-matched model (~3.8 ms/token, so 400 characters in
about 1.5 s).

ONNX Runtime would help the *large* model, where NumPy costs ~31 ms/token and
1000 characters take ~30 s -- too slow to demonstrate live. ONNX Runtime is
installed with a CoreML provider on Apple silicon, so the payoff should be real,
but the export is not yet correct and **must not be used until it is**:

    The cache-length axis is being frozen at the dummy value. The exported
    graph reports ``past_k: [n_layers, 1, 'past_len', 2, 64]`` -- the head axis
    (2) was marked dynamic instead of the length axis, so runtime rejects any
    cache whose length is not the dummy's. The legacy TorchScript path
    mis-binds ``dynamic_axes`` for this 5-D cache, and the dynamo path needs the
    head dimension restructured to a fixed size first.

Fixing it means reshaping the cache to put the length axis where the exporter
expects it and re-verifying against the NumPy engine before trusting it. Until
then, treat this file as a starting point, and keep the large model to short
generations (~300 characters, about 9 s) if you want to demo it.

Usage:
    python export_transformer_onnx.py models/tx-paired.h5
    python export_transformer_onnx.py models/*.h5 --out-dir models/onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vernebot.transformer import VerneTransformer
from vernebot.transformer_model import TransformerConfig, VerneTransformer as TorchTransformer


def build_torch_model_from_export(path: str | Path) -> TorchTransformer:
    """Recreate the exact PyTorch module from an exported HDF5 file."""
    import h5py
    import numpy as np

    engine = VerneTransformer.load(path)
    cfg = TransformerConfig(
        name=engine.config.name, vocab_size=engine.config.vocab_size,
        context=engine.config.context, d_model=engine.config.d_model,
        n_layers=engine.config.n_layers, n_heads=engine.config.n_heads,
        d_ff=engine.config.d_ff, dropout=0.0,
        tie_embeddings=False,          # weights are loaded explicitly below
        positional=engine.config.positional,
    )
    model = TorchTransformer(cfg)

    with h5py.File(path, "r") as f:
        def a(name: str) -> np.ndarray:
            return np.asarray(f[name], dtype=np.float32)

        sd = {
            "wte.weight": a("embedding/wte"),
            "lnf.weight": a("final_norm/weight"),
            "lnf.bias": a("final_norm/bias"),
            "head.weight": a("head/weight"),
        }
        if "embedding/wpe" in f:
            sd["wpe.weight"] = a("embedding/wpe")
        for i in range(cfg.n_layers):
            p = f"blocks/{i}"
            sd[f"blocks.{i}.ln1.weight"] = a(f"{p}/ln1/weight")
            sd[f"blocks.{i}.ln1.bias"] = a(f"{p}/ln1/bias")
            sd[f"blocks.{i}.attn.wq.weight"] = a(f"{p}/attn/wq/weight")
            sd[f"blocks.{i}.attn.wq.bias"] = a(f"{p}/attn/wq/bias")
            sd[f"blocks.{i}.attn.wk.weight"] = a(f"{p}/attn/wk/weight")
            sd[f"blocks.{i}.attn.wk.bias"] = a(f"{p}/attn/wk/bias")
            sd[f"blocks.{i}.attn.wv.weight"] = a(f"{p}/attn/wv/weight")
            sd[f"blocks.{i}.attn.wv.bias"] = a(f"{p}/attn/wv/bias")
            sd[f"blocks.{i}.attn.wo.weight"] = a(f"{p}/attn/wo/weight")
            sd[f"blocks.{i}.attn.wo.bias"] = a(f"{p}/attn/wo/bias")
            sd[f"blocks.{i}.ln2.weight"] = a(f"{p}/ln2/weight")
            sd[f"blocks.{i}.ln2.bias"] = a(f"{p}/ln2/bias")
            sd[f"blocks.{i}.mlp.fc.weight"] = a(f"{p}/mlp/fc/weight")
            sd[f"blocks.{i}.mlp.fc.bias"] = a(f"{p}/mlp/fc/bias")
            sd[f"blocks.{i}.mlp.proj.weight"] = a(f"{p}/mlp/proj/weight")
            sd[f"blocks.{i}.mlp.proj.bias"] = a(f"{p}/mlp/proj/bias")

    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in sd.items()}, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected weights: {unexpected}")
    model.eval()
    return model


class OnnxWrapper(torch.nn.Module):
    """ONNX-friendly forward: explicit past KV tensors in, present KV out.

    ``torch.onnx.export`` cannot trace a HuggingFace-style cache object, so the
    cache is passed as plain tensors here.
    """

    def __init__(self, model: TorchTransformer) -> None:
        super().__init__()
        self.model = model
        self.cfg = model.cfg

    def forward(self, tokens, past_k, past_v):
        """``past_k``/``past_v`` are (n_layers, batch, past_len, n_heads, d_head).

        Keeping the cache length on axis 2 (the conventional layout) means the
        single dynamic axis in ``dynamic_axes`` is the length for both the
        inputs and the outputs -- marking the head axis instead silently froze
        the cache length at the dummy value.
        """
        cfg = self.cfg
        T = tokens.shape[1]
        past_len = past_k.shape[2]
        pos = torch.arange(past_len, past_len + T, device=tokens.device)

        x = self.model.wte(tokens)
        if self.model.wpe is not None:
            x = x + self.model.wpe(pos)
        else:
            x = x + self.model._sinusoidal(pos, x.dtype)

        present_k, present_v = [], []
        for i, block in enumerate(self.model.blocks):
            h = block.ln1(x)
            # (T, H, D) -> (H, T, D) so that concatenating along the length axis
            # yields (H, past_len + T, D).
            def split(t):
                return t.view(T, cfg.n_heads, cfg.d_head).transpose(0, 1)

            q = split(block.attn.wq(h))
            k = split(block.attn.wk(h))
            v = split(block.attn.wv(h))

            full_k = torch.cat([past_k[i], k], dim=1)
            full_v = torch.cat([past_v[i], v], dim=1)
            present_k.append(full_k)
            present_v.append(full_v)

            # The engine only ever uses two shapes: the whole prompt at once
            # (past_len == 0, T > 1) or a single new token (T == 1). An explicit
            # causal mask keeps both correct and exports cleanly, unlike
            # ``is_causal``.
            total = past_len + T
            causal = torch.ones(T, total, dtype=torch.bool)
            if past_len == 0:
                causal = torch.tril(causal)
            att = torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(0), full_k.unsqueeze(0), full_v.unsqueeze(0),
                attn_mask=causal)
            att = att.squeeze(0).transpose(0, 1).contiguous().view(1, T, cfg.d_model)
            x = x + block.attn.wo(att)

            h = block.ln2(x)
            x = x + block.mlp.proj(torch.nn.functional.gelu(block.mlp.fc(h)))

        x = self.model.lnf(x)
        return self.model.head(x), torch.stack(present_k), torch.stack(present_v)


def export_one(h5_path: Path, out_dir: Path) -> Path:
    model = build_torch_model_from_export(h5_path)
    cfg = model.cfg
    wrapper = OnnxWrapper(model).eval()

    # (n_layers, batch, past_len, n_heads, d_head)
    past_shape = (cfg.n_layers, 1, 2, cfg.n_heads, cfg.d_head)
    dummy_tokens = torch.zeros((1, 2), dtype=torch.long)
    dummy_k = torch.zeros(past_shape, dtype=torch.float32)
    dummy_v = torch.zeros(past_shape, dtype=torch.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / (h5_path.stem + ".onnx")

    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_k, dummy_v),
        str(onnx_path),
        input_names=["tokens", "past_k", "past_v"],
        output_names=["logits", "present_k", "present_v"],
        opset_version=18,
        do_constant_folding=True,
        # Let the dynamo exporter infer the dynamic axes. Passing dynamic_axes
        # here caused the head axis to be frozen at the dummy value instead of
        # the cache length, which then rejected real cache sizes at runtime.
        dynamo=True,
    )
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export transformer(s) to ONNX.")
    parser.add_argument("paths", nargs="+", help="Exported .h5 model files.")
    parser.add_argument("--out-dir", default="models/onnx")
    args = parser.parse_args()

    for p in args.paths:
        path = Path(p)
        out = export_one(path, Path(args.out_dir))
        print(f"{path} -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
