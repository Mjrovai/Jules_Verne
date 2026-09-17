#!/bin/sh
# Two extra seeds per model, to see whether the RNN/Transformer gap is real.
set -e
cd "$(dirname "$0")/.."
for s in 2 3; do
  .venv/bin/python -u train_transformer.py --preset paired-ctx120 --steps 6000 --batch-size 64 \
      --lr 3e-3 --warmup 300 --eval-every 250 --sample-every 0 --device mps --seed "$s" \
      --out "models_v2/seeds/tx-paired-ctx120-seed$s.h5"
  rm -f "models_v2/seeds/tx-paired-ctx120-seed$s.h5"   # only the run record is needed
done
