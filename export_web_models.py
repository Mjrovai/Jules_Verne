#!/usr/bin/env python3
"""
Export the trained models for the browser as float16 binaries plus a manifest.

Why not just ship the ``.h5`` / ``.keras`` files?
-------------------------------------------------
Three reasons, all of them practical:

1. **Size.** The RNN is 4,095,867 parameters: 16.4 MB as float32, 8.2 MB as
   float16. ``tx-large.h5`` is 97 MB, which is over GitHub's 100 MB per-file hard
   limit and far too much to hand a browser.
2. **Format.** Reading HDF5 in JavaScript would mean shipping an HDF5 parser.
   Flat little-endian float16 blobs are read directly into a ``Float32Array``
   after a cheap bit-level conversion.
3. **Fidelity.** float16 keeps the weights small with no measurable change to the
   output, whereas int8 quantisation would need per-tensor scales and would
   change the sampled text -- which matters when the whole point is comparing two
   models on identical settings.

Output layout (all under ``--out``)::

    manifest.json          tensor names -> {file, shape, offset, dtype}
    rnn.bin                the GRU model's tensors, concatenated in order
    tx-paired.bin          the 4M Transformer, context 256
    tx-paired-ctx120.bin   the 4M Transformer, context 120 -- the one that
                           isolates architecture, since it matches the RNN's
                           window as well as its size

The web demo fetches the manifest, then one binary per model, and views them as
flat arrays.

**Orientation contract.** Tensors are exported in the orientation the JavaScript
expects, so the browser never transposes:

* GRU ``kernel_*`` and ``recurrent_*`` are ``(out, in)`` -- Keras stores them the
  other way round, so they are transposed here.
* ``dense_w`` stays ``(units, vocab)``, used as ``x @ W``.
* Transformer projections and MLP weights stay ``(out, in)``, as PyTorch stores
  them, used as ``W @ x``.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "docs" / "assets" / "models"

#: Templates whose ``{i}`` is replaced by the layer index.
TRANSFORMER_SPEC = {
    "wte": "embedding/wte",
    "wpe": "embedding/wpe",
    "lnf_g": "final_norm/weight",
    "lnf_b": "final_norm/bias",
    "head": "head/weight",
}
TRANSFORMER_LAYER_SPEC = {
    "ln1_g": "blocks/{i}/ln1/weight",
    "ln1_b": "blocks/{i}/ln1/bias",
    "wq_w": "blocks/{i}/attn/wq/weight",
    "wq_b": "blocks/{i}/attn/wq/bias",
    "wk_w": "blocks/{i}/attn/wk/weight",
    "wk_b": "blocks/{i}/attn/wk/bias",
    "wv_w": "blocks/{i}/attn/wv/weight",
    "wv_b": "blocks/{i}/attn/wv/bias",
    "wo_w": "blocks/{i}/attn/wo/weight",
    "wo_b": "blocks/{i}/attn/wo/bias",
    "ln2_g": "blocks/{i}/ln2/weight",
    "ln2_b": "blocks/{i}/ln2/bias",
    "fc_w": "blocks/{i}/mlp/fc/weight",
    "fc_b": "blocks/{i}/mlp/fc/bias",
    "proj_w": "blocks/{i}/mlp/proj/weight",
    "proj_b": "blocks/{i}/mlp/proj/bias",
}


def _to_fp16(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).astype(np.float16)


class Bundle:
    """Accumulates tensors and writes them as one contiguous float16 blob."""

    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.entries: dict[str, dict] = {}
        self.offset = 0

    def add(self, name: str, array: np.ndarray) -> None:
        flat = _to_fp16(array).reshape(-1)
        self.parts.append(flat)
        self.entries[name] = {
            "shape": list(array.shape),
            "offset": self.offset,
            "length": int(flat.size),
        }
        self.offset += int(flat.size)

    def write(self, path: Path) -> None:
        blob = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float16)
        # little-endian explicitly, so the file is platform-independent
        blob.astype("<f2").tofile(path)

    @property
    def params(self) -> int:
        return self.offset


# --------------------------------------------------------------------------
# RNN (.keras)
# --------------------------------------------------------------------------


def export_rnn(model_path: Path, bundle: Bundle) -> dict:
    """Read the GRU model's tensors out of the .keras zip.

    Tensor names match ``vernebot/engine.py`` so the JS port and the Python
    engine can be checked against each other key by key. The GRU kernel and
    recurrent kernel are split per gate here, at export time, so the JavaScript
    never has to know Keras' gate ordering.
    """
    with zipfile.ZipFile(model_path) as archive, archive.open("model.weights.h5") as raw:
        with h5py.File(raw, "r") as f:
            g = f["layers"]
            emb = g["embedding/vars/0"][:]
            kernel = g["gru/cell/vars/0"][:]          # (256, 3072)
            recurrent = g["gru/cell/vars/1"][:]       # (1024, 3072)
            bias = g["gru/cell/vars/2"][:]            # (2, 3072)
            dense_k = g["dense/vars/0"][:]            # (1024, 123)
            dense_b = g["dense/vars/1"][:]

    units = kernel.shape[1] // 3

    # ------------------------------------------------------------------
    # Orientation contract
    # ------------------------------------------------------------------
    # Everything the browser reads is stored as (out, in), so a linear layer is
    # always `y = W @ x`. Keras and PyTorch store the GRU and Dense weights the
    # other way round for this model's use, so they are transposed here -- once,
    # at export time -- rather than in JavaScript on every generated character.
    #
    # Getting this wrong is silent: a transposed matrix still has plausible
    #-looking numbers, and the mistake only shows up as text that is subtly or
    # completely wrong.
    def gate(matrix: np.ndarray, which: int) -> np.ndarray:
        """One gate's weights, as (units, in)."""
        part = matrix[:, which * units:(which + 1) * units]     # (in, units)
        return np.ascontiguousarray(part.T)                     # (units, in)

    bundle.add("wte", emb)                                      # (vocab, d_model)
    bundle.add("kernel_z", gate(kernel, 0))                     # (units, d_model)
    bundle.add("kernel_r", gate(kernel, 1))
    bundle.add("kernel_h", gate(kernel, 2))
    bundle.add("recurrent_z", gate(recurrent, 0))               # (units, units)
    bundle.add("recurrent_r", gate(recurrent, 1))
    bundle.add("recurrent_h", gate(recurrent, 2))
    bundle.add("input_bias_z", bias[0, :units])
    bundle.add("input_bias_r", bias[0, units: 2 * units])
    bundle.add("input_bias_h", bias[0, 2 * units:])
    bundle.add("recurrent_bias_z", bias[1, :units])
    bundle.add("recurrent_bias_r", bias[1, units: 2 * units])
    bundle.add("recurrent_bias_h", bias[1, 2 * units:])
    bundle.add("dense_w", np.ascontiguousarray(dense_k.T))      # (vocab, units)
    bundle.add("dense_b", dense_b)

    return {
        "architecture": "rnn",
        "vocab_size": int(emb.shape[0]),
        "d_model": int(emb.shape[1]),
        "units": int(units),
        "context": 120,
    }


# --------------------------------------------------------------------------
# Transformer (.h5)
# --------------------------------------------------------------------------


def export_transformer(model_path: Path, bundle: Bundle) -> dict:
    """Read an exported Transformer, flattening the layer list into names."""
    with h5py.File(model_path, "r") as f:
        meta = json.loads(f.attrs["config"])
        cfg = meta["config"]

        for name, path in TRANSFORMER_SPEC.items():
            if path in f:
                bundle.add(name, f[path][:])
        for i in range(cfg["n_layers"]):
            for name, template in TRANSFORMER_LAYER_SPEC.items():
                bundle.add(f"{name}_{i}", f[template.format(i=i)][:])

        vocab = meta.get("vocab", "")

    return {
        "architecture": "transformer",
        "vocab_size": len(vocab) or int(cfg["vocab_size"]),
        "d_model": int(cfg["d_model"]),
        "n_layers": int(cfg["n_layers"]),
        "n_heads": int(cfg["n_heads"]),
        "d_ff": int(cfg["d_ff"]),
        "context": int(cfg["context"]),
        "positional": cfg.get("positional", "learned"),
        "parameter_count": int(meta.get("parameter_count", 0)),
        "training_loss": meta.get("training_loss"),
        "vocab": vocab,
    }


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def vocabulary() -> str:
    """The 123-character vocabulary, rebuilt exactly as training built it."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from vernebot.engine import build_vocabulary

    return "".join(build_vocabulary())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export models for the web demo.")
    parser.add_argument("--rnn", default=str(PROJECT_ROOT / "models" / "verne_rnn_model.keras"))
    parser.add_argument("--paired", default=str(PROJECT_ROOT / "models" / "tx-paired.h5"),
                        help="The size-matched Transformer, context 256. Skipped if missing.")
    parser.add_argument("--paired-ctx120",
                        default=str(PROJECT_ROOT / "models" / "tx-paired-ctx120.h5"),
                        help="The size-matched Transformer at the RNN's own window. "
                             "Skipped if missing.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--no-vocab", action="store_true",
                        help="Do not embed the vocabulary (reads books/ otherwise).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "format": "verne-fp16-v1",
        "dtype": "float16",
        "byteOrder": "little",
        "models": {},
    }

    if not args.no_vocab:
        vocab = vocabulary()
        if len(vocab) != 123:
            raise SystemExit(f"expected a 123-character vocabulary, got {len(vocab)}")
        manifest["vocab"] = vocab
        print(f"vocabulary: {len(vocab)} characters")

    # RNN
    rnn_path = Path(args.rnn)
    if rnn_path.exists():
        bundle = Bundle()
        info = export_rnn(rnn_path, bundle)
        bundle.write(out / "rnn.bin")
        info.update({
            "file": "rnn.bin",
            "tensors": bundle.entries,
            "parameter_count": bundle.params,
            "source": rnn_path.name,
        })
        manifest["models"]["rnn"] = info
        size = (out / "rnn.bin").stat().st_size
        print(f"rnn        : {bundle.params:>12,} params -> rnn.bin ({size / 1e6:.1f} MB)")
    else:
        print(f"rnn        : skipped ({rnn_path} not found)")

    # The Transformers. Both are exported: one matches the RNN's window and
    # isolates the architecture, the other shows what a longer window buys.
    for key, source in (("tx-paired", args.paired),
                        ("tx-paired-ctx120", args.paired_ctx120)):
        path = Path(source)
        if not path.exists():
            print(f"{key:17s}: skipped ({path} not found)")
            continue
        bundle = Bundle()
        info = export_transformer(path, bundle)
        bundle.write(out / f"{key}.bin")
        # ``bundle.params`` double-counts a weight-tied embedding, because the
        # output head is the same tensor and the exporter writes it under both
        # names. The authoritative figure is the one stored in the file, which
        # comes from the training script and counts each parameter once.
        info.update({
            "key": key,
            "file": f"{key}.bin",
            "tensors": bundle.entries,
            "parameter_count": info.get("parameter_count") or bundle.params,
            "exported_values": bundle.params,
            "source": path.name,
        })
        manifest["models"][key] = info
        size = (out / f"{key}.bin").stat().st_size
        print(f"{key:17s}: {bundle.params:>12,} params -> {key}.bin ({size / 1e6:.1f} MB)")

    if not manifest["models"]:
        raise SystemExit("no models exported")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))

    total = sum((out / m["file"]).stat().st_size for m in manifest["models"].values())
    print(f"\nmanifest   : {manifest_path} ({manifest_path.stat().st_size / 1024:.1f} KB)")
    print(f"total      : {total / 1e6:.1f} MB of weights")
    print("\nThe large Transformer is deliberately not exported: at 97 MB it exceeds")
    print("GitHub's per-file limit and is impractical for a browser download.")


if __name__ == "__main__":
    main()
