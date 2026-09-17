# The RNN vs Transformer Experiment

A guide to the comparison this project is built around, for use in the lecture.
It records the design decisions, the reasoning behind them, and the numbers
measured on the development machine.

---

## The question

The course's original notebook ends by naming three things that separate its
small character model from a modern LLM:

1. **Scale** — 4M parameters against 175B+
2. **Architecture** — recurrence against attention
3. **Tokenization** — characters against subwords

It would be easy, and wrong, to build a Transformer that changes all three at
once and then declare that transformers write better Verne. The result would be
uninterpretable: nobody could say which change did the work.

## The design

**Tokenization is held fixed. Architecture and scale are varied separately.**

Every model here uses the same character-level tokenizer: the 123 symbols
obtained by `sorted(set(text))` over the ten novels in `books/`. The training
script refuses to run if the vocabulary does not match.

Two decisions make the losses below comparable at all, and both were fixed after
the first round of runs:

**The corpus is cleaned.** `clean_corpus.py` strips the Project Gutenberg header
and license from every novel and writes `books_clean/` — 5,571,763 characters,
3.4% less than the raw 5,768,791. The license alone was enough for the original
RNN to learn to recite it. The vocabulary is still built from the raw `books/`,
so it stays at 123 characters and every model remains token-compatible; the
three symbols that occurred only in the boilerplate (`•`, `™`, the byte-order
mark) simply never appear.

**Validation is held out per book.** Each novel contributes a contiguous 5%
slice from its middle: 278,888 characters in total, spread across all ten books.
The earlier split took the last 5% of the joined corpus, which was the end of one
novel plus its license. No training window crosses a book boundary or a held-out
slice. The RNN is trained on exactly the same split by
`Train_JulesVerne_RNN.ipynb`, so for the first time both architectures report a
held-out number.

| Run | Parameters | vs RNN | Context | Varies |
|---|---|---|---|---|
| RNN (`rnn-split`) | 4,095,867 | 1.00x | 120 | baseline |
| `paired-ctx120` | 4,011,520 | 0.98x | 120 | **architecture only** |
| `paired` | 4,046,336 | 0.99x | 256 | architecture + context |
| `large` | 25,414,144 | 6.20x | 256 | **scale** |

The size match is the point of the first row. At a **2% parameter difference**
and an identical context window, neither model can win on size or on memory.
If the Transformer writes better, attention is the reason.

The pairs are then read as a sequence:

- `paired-ctx120` **vs RNN** → what attention buys at equal size and equal window
- `paired` **vs `paired-ctx120`** → what a longer window buys at equal size
- `large` **vs `paired`** → what scale buys at equal architecture

That decomposition is what makes each claim defensible on its own.

## Why not hold the tokenizer and compare against a big model?

Because it is the most common way this comparison goes wrong. A 30M-parameter
Transformer with a 66k word vocabulary would beat the RNN, and the natural
conclusion — "transformers are better" — would be unsupported: the win could
come entirely from the extra parameters.

Word-level tokenization also has a concrete cost here. The corpus has 66,453
word types, so the output layer alone would be `d_model × 66,453`. At
`d_model=256` that is **17M parameters in a single layer**, four times the
entire RNN. That is worth showing in class as the reason subword tokenization
exists, but as a separate lesson, not folded into this one.

## Training results

The three size-matched models were retrained on the cleaned corpus with the
per-book split; each was run with **three random seeds**. Every run exports the
checkpoint with the lowest validation loss, not the last step. Run records are in
`models_v2/`, and the weights beside them.

| Run | Parameters | vs RNN | Context | Best checkpoint | Time | Train loss | **Val loss** | Perplexity |
|---|---|---|---|---|---|---|---|---|
| RNN (`rnn-split`) | 4,095,867 | 1.00x | 120 | epoch 10 of 30 | 91 min (CPU) | 1.008 | **1.146** | 3.15 |
| `tx-paired-ctx120` | 4,011,520 | 0.98x | 120 | step 6,000 of 6,000 | 5.7 min (GPU) | 1.054 | **1.102** | 3.01 |
| `tx-paired` | 4,046,336 | 0.99x | 256 | step 6,000 of 6,000 | 12.1 min (GPU) | 0.959 | **1.034** | 2.81 |

Seed 1337 is shown. Across three seeds:

| Model | Seed 1337 | Seed 2 | Seed 3 | Mean | Std dev |
|---|---|---|---|---|---|
| RNN | 1.1460 | 1.1475 | 1.1497 | **1.1477** | 0.0019 |
| `tx-paired-ctx120` | 1.1019 | 1.1057 | 1.1046 | **1.1041** | 0.0020 |

All three are in the browser demo at
<https://mjrovai.github.io/Jules_Verne/>, where **Run all three** writes from one
seed with each of them, so the losses above can be read next to the prose they
correspond to.

`tx-large` has **not** been retrained on this corpus and split. Its numbers from
the first round (val 1.083 at the last step, 0.998 at its best step) were
measured on different data and must not be put in the same table.

Timings are an Apple M5 Max: the Transformers on the GPU through PyTorch MPS, the
RNN on the CPU because TensorFlow has no Metal path in this environment. A Colab
T4 trains the RNN in roughly a third of that.

### Reading these numbers honestly

**At equal size and equal window, attention wins — by a little.** The
size-matched Transformer reaches 1.104 against the RNN's 1.148, a 4% reduction,
on text neither model saw. The gap is 0.044, about 22 times the seed-to-seed
standard deviation of either model, so it is not noise. Both models had seen a
comparable amount of text at their best checkpoint: 52M characters for the RNN,
46M for the Transformer.

**Context length buys more than the architecture change does.** Going from a
120- to a 256-character window at the same size improves held-out loss from
1.102 to 1.034 — a further 0.068, larger than the 0.044 that attention itself
bought. *At fixed size, more context buys accuracy.*

**The RNN memorizes, and 30 epochs is past the point of usefulness.** Its
validation loss bottoms out at epoch 10 in all three seeds and then rises to
1.245 by epoch 30 while training loss keeps falling to 0.869. This is what the
original notebook's 0.45 was measuring: the training set, at the end of a run
that had long since started overfitting. Without the best-checkpoint callback,
the exported model would be the worst one of the run.

**One caveat weakens the architecture claim.** The Transformers use dropout 0.1
and the course RNN uses none, so part of the 0.044 may be regularization rather
than attention. Training the RNN with dropout would close that hole; it has not
been done.

**Validation loss is not writing quality.** All three write plausible Verne at
temperature 0.7, and none of them recite the Gutenberg license any more.

### Coherence used to decay with length, and the cause was the cache

Every Transformer model used to drift after a few hundred characters: coherent
prose for roughly 200–250 characters, then word salad. The cause was the KV
cache, not the models, and the earlier claim here — that the ring cache had been
ruled out — was wrong.

**What was wrong.** With learned absolute positions, a cached token carries the
position it was written with. Once the ring wrapped, the newest character was
written into the oldest slot and so carried a *low* position, while older text
kept the high ones. The window the model read was a cyclic rotation of the real
text. The check that missed this compared the ring against exact replay only
*within the first window*, where the two agree by construction.

It shows up plainly once replay is run past the window, same weights, same seed:

| | last characters of 400, `tx-paired` |
|---|---|
| Ring cache | `theanoustheleathain peareagrexthe sourilleasirerathisousthe` |
| Exact replay | `the _Halbrane_ continued her life at the vessel to present her down to the sea` |

**The fix.** The cache is never allowed to wrap. Generation appends, which is
exact, and when the window fills the cache is rebuilt: the oldest 64 characters
are dropped and the rest are recomputed at their new positions, freeing 64 slots
to append into. That is one window pass per 64 characters instead of one per
character.

| `tx-paired`, 600 characters | ms per character | Real words past the window |
|---|---|---|
| Ring, wrapping (old) | 3.5 | ~50% |
| Rebuilt every 64 characters | 9.5 | ~98% |
| Full replay | ~330 | ~98% |

The cost is 1.7x for the 120-character model and 2.7x for the 256-character one.
`tests/check_js_parity.mjs` now measures this rather than trusting it: it
generates past the window and fails below 85% real words, which the old
behaviour does not reach.

**What this changes in the lecture.** The drift was an implementation bug, not a
property of small Transformers. Model capacity and context length may still
limit coherence further out, but that is now an open question, not a measured
result.

## Cost summary

The RNN's cost per character is constant and tiny. The Transformers pay for
attention at every character, plus one window rebuild per 64 characters once the
text is longer than the window, which is why the web app caps them at 1,500
characters per request and the RNN at 4,000. At 8 ms per character, a 1,500-
character request from `tx-paired` takes about 12 s. The numbers are in
"Measured inference cost" below.

### Training cost

Timed on an Apple M5 Max; a Colab T4 is broadly comparable for models this
small.

| Run | s/step | 4,000 steps | 6,000 steps | 12,000 steps |
|---|---|---|---|---|
| paired ctx120, batch 64 | 0.30 | 20 min | 30 min | 59 min |
| paired ctx256, batch 64 | 0.40 | 27 min | 40 min | 81 min |
| large ctx256, batch 48 | 1.31 | 88 min | 131 min | 263 min |

The two paired runs together are about **70 minutes** and carry the main claim.
The large run adds the scale story for roughly four more hours; it is the
optional part.

## Correctness of the inference engine

The website does not use PyTorch or TensorFlow. `vernebot/transformer.py`
reimplements the forward pass in NumPy and reads the exported HDF5 weights. It
was verified against PyTorch before any training was run:

| Check | Result |
|---|---|
| Max logit difference, all positions | **8 × 10⁻⁸** (float32 round-off) |
| Greedy generation agreement, 123 tokens | **123 / 123 = 100%** |
| Round trip PyTorch → HDF5 → NumPy | exact |

One subtlety is worth putting on a slide, because it is a real architectural
difference rather than an implementation detail:

> A KV cache is only valid while the context window does not move. The keys and
> values for a past token were computed **in the context of everything before
> it**, so once generation passes the window length and the window slides, the
> cached entries no longer describe what a fresh forward pass would compute.

With learned absolute positions there is no cheap way around that: a cached
token carries the position it was written with, so a wrapping ring rotates the
window and the output collapses (see "Coherence used to decay with length"
above). The engine therefore appends while there is room, and rebuilds when the
window fills — dropping the oldest 64 characters and recomputing the rest.

| Approach, `tx-paired`, 600 characters | Cost | Coherent? |
|---|---|---|
| Replay the window each character | ~330 ms/char | yes |
| Rebuild every 64 characters | 9.5 ms/char | yes |
| Wrapping ring | 3.5 ms/char | **no** |

Two details are load-bearing, and both were bugs first:

1. **The cache must not wrap.** `slot = pos % context` is only the token's
   position while the window is still filling. Once it wraps, that same
   arithmetic assigns the newest character a low position, and the model reads a
   rotated window.
2. **The prompt is capped at `context - 1` characters**, reserving a slot so the
   first generated token never lands on a position a prompt token already used.

Rotary or relative positions (RoPE, ALiBi) would make a true sliding window
valid and remove the rebuild entirely. That is the principled fix, and it needs
retraining.

This contrasts usefully with the RNN, whose hidden state *is* a valid summary of
all history and therefore needs no such correction.

### Measured inference cost

400 characters, seed `THE FLYING SUBMARINE`, temperature 0.7, pure NumPy, with
the cache rebuilt every 64 characters:

| Model | Context | ms per character | 400 chars |
|---|---|---|---|
| RNN, hidden state carried | 120 | 0.15 | 0.06 s |
| Transformer matched | 120 | 5.80 | 2.32 s |
| Transformer matched | 256 | 8.08 | 3.23 s |
| Transformer large | 256 | 66.39 | 26.56 s |

The RNN remains ~40x faster per character than the size-matched Transformer, and
~440x faster than the 25M one. That is the O(1) versus O(C) distinction with real
numbers attached. The rebuild costs the Transformers 1.7x (context 120) to 2.7x
(context 256) against the old wrapping ring, which was faster and wrong.

## What the models are

Both are deliberately plain, so the lecture can point at every part:

**RNN** (from the original notebook): `Embedding(123→256)` → `GRU(1024,
reset_after=True)` → `Dense(123)`, trained on fixed 120-character windows.

**Transformer**: decoder-only, pre-norm, learned absolute positions, GELU MLP,
weight-tied output head. `d_model=256`, 5 layers, 4 heads, `d_ff=1024`,
dropout 0.1, AdamW at `lr=3e-3` with 300-step warmup, cosine decay, gradient
clipping at 1.0.

## Reproducing

```bash
# Colab (recommended): open Train_JulesVerne_Transformer.ipynb
# Locally, without a GPU:
python train_transformer.py --preset paired --context 120 --steps 6000 \
       --out models/tx-paired-ctx120.h5
```

Then serve everything, including the RNN:

```bash
python run.py
```

The app discovers every `*.keras` and `*.h5` in `models/` and offers them in the
model selector, with a "Compare all models" button that runs the same seed and
temperature across all of them at once.

## Limitations to state plainly

- **The 0.45 in the course notebook is a training loss** at epoch 30 of a run
  with no validation set. It is not comparable to anything here. The RNN's
  held-out number is **1.146**, from `Train_JulesVerne_RNN.ipynb`, which retrains
  the same architecture on the shared split.
- **The RNN has no dropout and the Transformers have 0.1.** Everything else is
  matched, but this is not, so the architecture comparison is not airtight.
- **Validation loss is not the same as writing quality.** Perplexity is a proxy;
  the side-by-side output is the thing to actually read in class.
- **The corpus is fixed and small** (5.57M characters after cleaning, ten
  novels). Conclusions
  about context length or scale at this size may not hold at larger sizes. That
  caveat is itself a useful teaching point.
- **The large model is slow to demonstrate in NumPy.** If the size story matters
  more than the transparency story, an ONNX Runtime path
  (`export_transformer_onnx.py`) is started but not yet correct; see the status
  note in that file.
