#!/bin/sh
# Retrain the two size-matched Transformers: books_clean/, per-book split, best checkpoint.
set -e
cd "$(dirname "$0")/.."
for spec in "paired-ctx120 tx-paired-ctx120" "paired tx-paired"; do
  set -- $spec
  echo "=== $2 started $(date)"
  .venv/bin/python -u train_transformer.py --preset "$1" --steps 6000 --batch-size 64 \
      --lr 3e-3 --warmup 300 --eval-every 250 --sample-every 2000 --device mps \
      --out "models_v2/$2.h5"
  echo "=== $2 finished $(date)"
done
