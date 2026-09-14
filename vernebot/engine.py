"""
Pure NumPy inference engine for the Jules Verne character-level RNN.

This module reimplements the exact forward pass of the Keras model built in
``20_JulesVerneBot_Generating_English_Texts_with_RNN.ipynb``:

    Embedding(vocab_size=123, embed_dim=256)
    GRU(rnn_neurons=1024, return_sequences=True, reset_after=True)
    Dense(vocab_size=123, activation="linear")   # logits

Why not just use TensorFlow/Keras?
----------------------------------
The trained weights are stored in plain HDF5, and the network is small
(~4.2M parameters). Reimplementing three layers in NumPy means the web app
needs no TensorFlow install, starts in under a second, and runs each
generation step in well under a millisecond. It also makes the arithmetic
visible for teaching, which is the point of the course.

The one subtlety is Keras' ``reset_after=True`` GRU ordering, documented
inline in :meth:`VerneRNN.gru_step`.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np

# --------------------------------------------------------------------------
# Paths / hyper-parameters (must match the trained model)
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJECT_ROOT / "books"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "verne_rnn_model.keras"

EMBED_DIM = 256
RNN_NEURONS = 1024
CONTEXT_LENGTH = 120  # the model was built with a fixed (1, 120) input shape

# Text generation defaults, mirroring the notebook
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_GENERATE = 800


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def build_vocabulary(books_dir: Path | str = BOOKS_DIR) -> list[str]:
    """Rebuild the character vocabulary exactly as the notebook did.

    The notebook concatenated every ``*.txt`` in ``books/`` and then took
    ``sorted(set(text))``. A set is order-independent, so this reproduces the
    training-time mapping as long as the book files are unchanged.
    """
    books_dir = Path(books_dir)
    if not books_dir.is_dir():
        raise FileNotFoundError(f"Books directory not found: {books_dir}")

    text_parts: list[str] = []
    for filename in sorted(os.listdir(books_dir)):
        if filename.endswith(".txt"):
            with open(books_dir / filename, "r", encoding="utf-8") as fh:
                text_parts.append(fh.read() + "\n")

    if not text_parts:
        raise FileNotFoundError(f"No .txt files found in {books_dir}")

    return sorted(set("".join(text_parts)))


# --------------------------------------------------------------------------
# Weight loading
# --------------------------------------------------------------------------


@dataclass
class GRUWeights:
    """Keras GRU weights, pre-split per gate.

    Gate order in Keras is ``z`` (update), ``r`` (reset), ``h`` (candidate).
    With ``reset_after=True`` Keras stores the bias as a ``(2, 3*units)``
    array: row 0 is the input bias and row 1 the recurrent bias, each split
    across the three gates.
    """

    kernel_z: np.ndarray
    kernel_r: np.ndarray
    kernel_h: np.ndarray
    recurrent_z: np.ndarray
    recurrent_r: np.ndarray
    recurrent_h: np.ndarray
    input_bias_z: np.ndarray
    input_bias_r: np.ndarray
    input_bias_h: np.ndarray
    recurrent_bias_z: np.ndarray
    recurrent_bias_r: np.ndarray
    recurrent_bias_h: np.ndarray


@dataclass
class VerneRNN:
    """The trained Jules Verne character RNN, ready for NumPy inference."""

    embedding: np.ndarray
    gru: GRUWeights
    dense_kernel: np.ndarray
    dense_bias: np.ndarray
    vocab: list[str]
    model_path: Path | None = None
    _char_to_idx: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        # Pre-transpose the embedding for fast row lookups.
        self.embedding = np.ascontiguousarray(self.embedding, dtype=np.float32)

    # -- properties --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def char_to_idx(self) -> dict[str, int]:
        return self._char_to_idx

    @property
    def idx_to_char(self) -> list[str]:
        return self.vocab

    @property
    def parameter_count(self) -> int:
        return (
            self.embedding.size
            + sum(
                w.size
                for w in (
                    self.gru.kernel_z,
                    self.gru.kernel_r,
                    self.gru.kernel_h,
                    self.gru.recurrent_z,
                    self.gru.recurrent_r,
                    self.gru.recurrent_h,
                    self.gru.input_bias_z,
                    self.gru.input_bias_r,
                    self.gru.input_bias_h,
                    self.gru.recurrent_bias_z,
                    self.gru.recurrent_bias_r,
                    self.gru.recurrent_bias_h,
                )
            )
            + self.dense_kernel.size
            + self.dense_bias.size
        )

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        books_dir: Path | str = BOOKS_DIR,
        vocab: Sequence[str] | None = None,
    ) -> "VerneRNN":
        """Load the ``.keras`` archive.

        A ``.keras`` file is a zip containing ``model.weights.h5``; only the
        ``layers/...`` group is read, so the Adam optimizer state is skipped
        and loading stays fast.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run `python export_model.py` from the notebook folder, or point "
                "VERNE_MODEL_PATH at your .keras file."
            )

        vocab_list = list(vocab) if vocab is not None else build_vocabulary(books_dir)

        with zipfile.ZipFile(model_path) as archive:
            with archive.open("model.weights.h5") as raw:
                # h5py needs a seekable file object; the zip entry is seekable
                # for stored/deflated entries on CPython.
                with h5py.File(raw, "r") as h5:
                    g = h5["layers"]

                    embedding = g["embedding/vars/0"][:].astype(np.float32)

                    gru_kernel = g["gru/cell/vars/0"][:].astype(np.float32)
                    gru_recurrent = g["gru/cell/vars/1"][:].astype(np.float32)
                    gru_bias = g["gru/cell/vars/2"][:].astype(np.float32)

                    dense_kernel = g["dense/vars/0"][:].astype(np.float32)
                    dense_bias = g["dense/vars/1"][:].astype(np.float32)

        units = gru_kernel.shape[1] // 3
        gru = GRUWeights(
            kernel_z=gru_kernel[:, :units],
            kernel_r=gru_kernel[:, units : 2 * units],
            kernel_h=gru_kernel[:, 2 * units :],
            recurrent_z=gru_recurrent[:, :units],
            recurrent_r=gru_recurrent[:, units : 2 * units],
            recurrent_h=gru_recurrent[:, 2 * units :],
            # With reset_after=True, Keras stores bias as (2, 3*units):
            # row 0 is the input bias, row 1 the recurrent bias.
            input_bias_z=gru_bias[0, :units],
            input_bias_r=gru_bias[0, units : 2 * units],
            input_bias_h=gru_bias[0, 2 * units :],
            recurrent_bias_z=gru_bias[1, :units],
            recurrent_bias_r=gru_bias[1, units : 2 * units],
            recurrent_bias_h=gru_bias[1, 2 * units :],
        )

        model = cls(
            embedding=embedding,
            gru=gru,
            dense_kernel=dense_kernel,
            dense_bias=dense_bias,
            vocab=vocab_list,
            model_path=model_path,
        )

        # Fail loudly on an architecture/vocabulary mismatch rather than
        # silently generating nonsense.
        if embedding.shape != (len(vocab_list), EMBED_DIM):
            raise ValueError(
                "Vocabulary/model mismatch: the model expects "
                f"{embedding.shape[0]} characters but {len(vocab_list)} were "
                "derived from books/. Check that books/ is unchanged since "
                "training, or set VERNE_VOCAB_PATH to a saved vocabulary."
            )
        if dense_bias.shape[0] != len(vocab_list):
            raise ValueError(
                f"Dense layer outputs {dense_bias.shape[0]} classes but the "
                f"vocabulary has {len(vocab_list)} characters."
            )

        return model

    # -- forward pass ------------------------------------------------------

    def gru_step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """One GRU timestep, matching Keras' ``reset_after=True`` semantics.

        Keras computes (see ``keras/src/layers/rnn/gru.py``, implementation 1)::

            z = sigmoid(x @ Wz + h @ Rz + bz + rbz)
            r = sigmoid(x @ Wr + h @ Rr + br + rbr)

            # reset_after=True: the reset gate multiplies the *result* of the
            # recurrent matmul, not the previous state going into it
            recurrent_h = r * (h @ Rh + rbh)

            hh = tanh(x @ Wh + bh + recurrent_h)

            h' = z * h + (1 - z) * hh

        Note that both the input bias ``b*`` and the recurrent bias ``rb*``
        apply to all three gates.
        """
        g = self.gru
        z = _sigmoid(x @ g.kernel_z + h @ g.recurrent_z + g.input_bias_z + g.recurrent_bias_z)
        r = _sigmoid(x @ g.kernel_r + h @ g.recurrent_r + g.input_bias_r + g.recurrent_bias_r)
        recurrent_h = r * (h @ g.recurrent_h + g.recurrent_bias_h)
        hh = np.tanh(x @ g.kernel_h + g.input_bias_h + recurrent_h)
        return z * h + (1.0 - z) * hh

    def logits(self, indices: Sequence[int], state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run the network over a character-index sequence.

        Returns ``(logits, final_hidden_state)`` where ``logits`` has shape
        ``(vocab_size,)`` and corresponds to the prediction after the last
        input character.
        """
        h = np.zeros(RNN_NEURONS, dtype=np.float32) if state is None else state
        emb = self.embedding
        for idx in indices:
            h = self.gru_step(emb[idx], h)
        return self.dense_kernel.T @ h + self.dense_bias, h

    def warm_state(self, text: str) -> tuple[np.ndarray, str]:
        """Build the GRU hidden state for a seed string.

        Only the last ``CONTEXT_LENGTH`` characters are consumed, because the
        model was trained on fixed 120-character windows and never saw longer
        context. Characters the model cannot represent (e.g. accented letters
        outside the vocabulary) are reported back so the caller can sanitise.
        """
        cleaned = "".join(c for c in text if c in self._char_to_idx)
        window = cleaned[-CONTEXT_LENGTH:]
        if not window:
            return np.zeros(RNN_NEURONS, dtype=np.float32), cleaned
        _, state = self.logits([self._char_to_idx[c] for c in window])
        return state, cleaned

    # -- sampling ----------------------------------------------------------

    def next_token(
        self,
        state: np.ndarray,
        last_index: int,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int | None = None,
        greedy: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, np.ndarray]:
        """Advance one character and sample the next index."""
        logits, state = self.logits([last_index], state=state)

        if greedy:
            return int(np.argmax(logits)), state

        temperature = max(float(temperature), 1e-4)
        logits = logits.astype(np.float64) / temperature

        if top_k is not None and 0 < top_k < logits.shape[0]:
            # Keep only the k highest logits, masking the rest to -inf.
            keep = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[keep] = logits[keep]
            logits = mask

        logits -= logits.max()
        probs = np.exp(logits)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return int(np.argmax(logits)), state
        probs /= total

        rng = rng or np.random.default_rng()
        return int(rng.choice(probs.shape[0], p=probs)), state

    # -- public generation API --------------------------------------------

    def generate(
        self,
        seed_text: str,
        num_generate: int = DEFAULT_NUM_GENERATE,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int | None = None,
        greedy: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterator[str]:
        """Yield generated characters one at a time.

        The seed text is returned as the first chunk so callers can simply
        concatenate everything they receive.
        """
        cleaned = "".join(c for c in seed_text if c in self._char_to_idx)
        if not cleaned:
            raise ValueError("Seed text must contain at least one in-vocabulary character.")

        rng = rng or np.random.default_rng()

        # Warm up on the tail of the seed, matching the 120-character windows
        # the model was trained on.
        state = np.zeros(RNN_NEURONS, dtype=np.float32)
        window = [self._char_to_idx[c] for c in cleaned[-CONTEXT_LENGTH:]]
        for idx in window:
            state = self.gru_step(self.embedding[idx], state)

        yield cleaned

        # Then feed the hidden state forward one character at a time. The GRU
        # state is its summary of everything read so far, so each new character
        # costs a single timestep (~0.1 ms) rather than a full context replay.
        # ``window`` is tracked only so the effective context stays at 120
        # characters, matching the training windows.
        last_index = window[-1]
        for _ in range(int(num_generate)):
            index, state = self.next_token(
                state,
                last_index,
                temperature=temperature,
                top_k=top_k,
                greedy=greedy,
                rng=rng,
            )
            yield self.vocab[index]

            window.append(index)
            if len(window) > CONTEXT_LENGTH:
                window.pop(0)
            last_index = index


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out
