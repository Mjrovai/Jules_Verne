"""One RNN seed, for the seed-variation check.

Same data, model and optimizer as Train_JulesVerne_RNN.ipynb, but it stops early
(the best epoch was 10 of 30) and records only the loss curve -- no weights.

    python models_v2/train_rnn_seed.py 2
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vernebot.engine import build_vocabulary
from train_transformer import load_books, split_by_book, encode

SEQ_LEN, BATCH_SIZE, MAX_EPOCHS, PATIENCE = 120, 128, 16, 3
seed = int(sys.argv[1])

vocab = build_vocabulary("books")
train_segments, val_segments = split_by_book(load_books("books_clean"), 0.05)


def to_sequences(segments):
    chunks = []
    for _, text in segments:
        ids = encode(text, vocab).astype(np.int32)
        n = len(ids) // (SEQ_LEN + 1)
        if n:
            chunks.append(ids[: n * (SEQ_LEN + 1)].reshape(n, SEQ_LEN + 1))
    return np.concatenate(chunks)


train_seq, val_seq = to_sequences(train_segments), to_sequences(val_segments)
train_ds = (tf.data.Dataset.from_tensor_slices((train_seq[:, :-1], train_seq[:, 1:]))
            .shuffle(len(train_seq)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
val_ds = (tf.data.Dataset.from_tensor_slices((val_seq[:, :-1], val_seq[:, 1:]))
          .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

keras.utils.set_random_seed(seed)
model = keras.Sequential([
    keras.Input(shape=(None,), dtype="int32"),
    keras.layers.Embedding(len(vocab), 256, name="embedding"),
    keras.layers.GRU(1024, return_sequences=True,
                     recurrent_initializer="glorot_uniform", name="gru"),
    keras.layers.Dense(len(vocab), name="dense"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

started = time.time()
h = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]).history
best = int(np.argmin(h["val_loss"]))
record = {
    "model": "rnn", "seed": seed, "best_epoch": best + 1, "epochs_run": len(h["loss"]),
    "best_val_loss": float(h["val_loss"][best]), "train_loss_at_best": float(h["loss"][best]),
    "train_minutes": round((time.time() - started) / 60, 2),
    "history": [{"epoch": e + 1, "train": float(h["loss"][e]), "val": float(h["val_loss"][e])}
                for e in range(len(h["loss"]))],
}
Path(f"models_v2/seeds/rnn-seed{seed}.json").write_text(json.dumps(record, indent=2))
print(f"seed {seed}: best val {record['best_val_loss']:.4f} at epoch {best + 1}")
