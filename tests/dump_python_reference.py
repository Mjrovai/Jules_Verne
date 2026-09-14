#!/usr/bin/env python3
"""
Dump reference values from the Python engines for the JavaScript parity test.

Greedy text is a poor check: the two implementations consume different precision
(the browser gets float16 weights), so a single flipped near-tie changes every
character after it. Comparing *numbers* instead isolates the port from the
quantisation.

Writes tests/reference.json with, for a fixed seed string:
  * the logits after the whole prompt, from each engine
  * the top-8 character indices, so a mismatch names the characters involved

Run this after retraining, then run ``node tests/check_js_parity.mjs``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPT = "THE FLYING SUBMARINE"
OUT = Path(__file__).resolve().parent / "reference.json"


def main() -> None:
    from vernebot.engine import VerneRNN
    from vernebot.transformer import VerneTransformer

    reference: dict = {"prompt": PROMPT, "models": {}}

    rnn = VerneRNN.load()
    logits, _state = rnn.logits([rnn.char_to_idx[c] for c in PROMPT])
    top = np.argsort(logits)[-8:][::-1]
    reference["models"]["rnn"] = {
        "logits": [float(v) for v in logits],
        "top": [int(i) for i in top],
        "top_chars": [rnn.vocab[int(i)] for i in top],
    }
    print(f"rnn       : logits {logits.shape}, top = {[rnn.vocab[int(i)] for i in top]}")

    tx_path = ROOT / "models" / "tx-paired.h5"
    if tx_path.exists():
        tx = VerneTransformer.load(tx_path)
        # The prompt fills the cache exactly as generation does, so this is the
        # same first-step decision the page makes.
        prompt = [tx.char_to_idx[c] for c in PROMPT[-(tx.config.context - 1):]]
        logits = tx.logits(prompt)
        top = np.argsort(logits)[-8:][::-1]
        reference["models"]["tx-paired"] = {
            "logits": [float(v) for v in logits],
            "top": [int(i) for i in top],
            "top_chars": [tx.vocab[int(i)] for i in top],
            "prompt_length": len(prompt),
        }
        print(f"tx-paired : logits {logits.shape}, top = {[tx.vocab[int(i)] for i in top]}")
    else:
        print("tx-paired : skipped (models/tx-paired.h5 not found)")

    OUT.write_text(json.dumps(reference))
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
