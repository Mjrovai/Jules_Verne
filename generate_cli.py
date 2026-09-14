#!/usr/bin/env python3
"""
Generate Jules Verne text from the command line, without the web UI.

Useful for quick checks, for scripting classroom examples, or for producing
reference output to paste into slides.

    python generate_cli.py "THE FLYING SUBMARINE"
    python generate_cli.py "CAPTAIN NEMO" --temperature 0.5 --length 1500
    python generate_cli.py "THE MOON" --greedy
    python generate_cli.py "MY UNCLE" --temperatures 0.5 0.7 1.0 --length 400
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from vernebot import DEFAULT_NUM_GENERATE, DEFAULT_TEMPERATURE, VerneRNN, reflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the Jules Verne RNN.")
    parser.add_argument("seed", help="Seed word or phrase.")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE}).")
    parser.add_argument("--temperatures", type=float, nargs="+",
                        help="Generate once per temperature, for comparison.")
    parser.add_argument("-n", "--length", type=int, default=DEFAULT_NUM_GENERATE,
                        help=f"Characters to generate (default {DEFAULT_NUM_GENERATE}).")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Restrict sampling to the k most likely characters.")
    parser.add_argument("--greedy", action="store_true",
                        help="Always take the most likely character (ignores temperature).")
    parser.add_argument("--seed-value", type=int, default=None,
                        help="RNG seed, for reproducible output.")
    parser.add_argument("--raw", action="store_true",
                        help="Show the raw character stream, keeping the model's hard "
                             "line wrapping (which mirrors the Gutenberg editions).")
    parser.add_argument("--quiet-stats", action="store_true", help="Do not print timing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = VerneRNN.load()
    if not args.quiet_stats:
        print(
            f"[model: {model.model_path.name} | vocab {model.vocab_size} | "
            f"{model.parameter_count:,} parameters]",
            file=sys.stderr,
        )

    temperatures = args.temperatures or [args.temperature]

    for temperature in temperatures:
        started = time.perf_counter()
        rng = np.random.default_rng(args.seed_value)
        text = "".join(
            model.generate(
                args.seed,
                num_generate=args.length,
                temperature=temperature,
                top_k=args.top_k,
                greedy=args.greedy,
                rng=rng,
            )
        )
        if not args.raw:
            text = reflow(text)
        elapsed = (time.perf_counter() - started) * 1000

        header = "greedy" if args.greedy else f"T = {temperature}"
        if not args.quiet_stats:
            print(f"\n===== {header}  ({elapsed:.0f} ms) =====", file=sys.stderr)
        print(text)
        if not args.quiet_stats and len(temperatures) > 1:
            print("-" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
