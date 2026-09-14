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

| Run | Parameters | vs RNN | Context | Varies |
|---|---|---|---|---|
| RNN (original) | 4,095,867 | 1.00x | 120 | baseline |
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

All three models were trained on a Colab GPU. Run records are in
`Train results/*.json`; the weights sit in `models/`.

| Run | Parameters | vs RNN | Context | Steps | Time | Train loss | **Val loss** | Perplexity |
|---|---|---|---|---|---|---|---|---|
| RNN (original) | 4,095,867 | 1.00x | 120 | — | — | 0.45 (train only) | — | — |
| `tx-paired-ctx120` | 4,011,520 | 0.98x | 120 | 6,000 | 9.2 min | 1.048 | **1.098** | 3.00 |
| `tx-paired` | 4,046,336 | 0.99x | 256 | 6,000 | 20.7 min | 0.941 | **1.009** | 2.74 |
| `tx-large` | 25,414,144 | 6.20x | 256 | 12,000 | 158.5 min | 0.597 | **1.083** | 2.95 |

### Reading these numbers honestly

**The size-matched pair answers the architecture question.** At a 2% parameter
difference and an identical 120-character window, the Transformer reaches a
held-out loss of **1.098**. The RNN's 0.45 is a *training* loss from the original
notebook, so the two are not like-for-like and no head-to-head claim should be
made from them.

**Longer context helps at equal size.** `tx-paired` (context 256) beats
`tx-paired-ctx120` (context 120) on held-out loss, 1.009 against 1.098, with the
same 5-layer, 4-head architecture and essentially the same parameter count. That
is the cleanest result here: *at fixed size, more context buys accuracy.*

**More parameters did not buy generalisation.** `tx-large` has 6.2x the
parameters and drove its **training** loss to 0.597 — far below the others — but
its **held-out** loss is 1.083, slightly *worse* than the 4M model at the same
context. The gap between 0.597 and 1.083 is overfitting, and it is visible in the
loss curve: from step 5,000 onward validation loss rises while training loss keeps
falling. A textbook demonstration that lowering training loss is not the goal.

**Qualitatively**, `tx-large` produces the most Verne-like prose — it emits
chapter headings and sustained descriptive sentences. But its training loss is
much better than its generalisation, so the fluent surface partly reflects
memorisation of the corpus.

### Coherence decays with length, and the cause is not established

All Transformer models drift after a few hundred characters: `tx-paired`
produces convincing dialogue for roughly 200–250 characters and then degrades
into word salad. The RNN degrades differently — it repeats Project Gutenberg
licence boilerplate, which is 3.3% of the corpus.

Two things were ruled out by measurement rather than assumption:

- **Not the ring-cache approximation.** Generation within the first window is
  byte-identical to exact replay (verified to 119 characters), and the
  degradation happens well before the window fills.
- **Not corpus contamination alone.** The Gutenberg headers and footers are only
  3.3% of the training text.

The likely causes are model capacity and context length, but that is a hypothesis.
The experiment that would settle it — retraining `tx-paired` at context 512, where
degradation should start around twice as far in — has **not** been run.

## Cost summary

The RNN's cost per character is constant and tiny. The Transformers pay for
attention at every character, which is why the web app caps them at 1,500
characters per request and the RNN at 4,000. The numbers are in
"Measured inference cost after the fix" above.

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

The engine treats the cache as a **ring**: each new character overwrites the
oldest slot and costs exactly one pass through the stack. The alternative —
replaying the window for every character — is exact but pathological, and this
was measured on the trained models rather than assumed:

| Approach | 367 characters, `tx-paired-ctx120` |
|---|---|
| Replay the window each character | **122 s** |
| Ring cache | **1.2 s** |

Before the fix, characters past the window cost ~340 ms each instead of ~3 ms,
which is invisible at 200 characters and catastrophic at 400. It made the
four-model comparison time out at 300 s; it now completes in 10.6 s.

Two details make the ring correct, and both were bugs first:

1. **`slot = pos % context` is the token's position in the current window.**
   Renumbering the window from its own start is what training saw. Using the
   absolute position instead assigned a generated token the position embedding of
   a much later slot, and the output collapsed into gibberish from the very first
   wrapped character.
2. **The prompt is capped at `context - 1` characters**, reserving a slot so the
   first generated token never lands on a position a prompt token already used.

The ring matches exact replay **character for character through the first
window** (verified: identical for the first 119 generated characters), then
diverges deliberately. That is the sliding-window approximation, and it is the
price of flat per-character cost.

This contrasts usefully with the RNN, whose hidden state *is* a valid summary of
all history and therefore needs no such correction.

### Measured inference cost after the fix

400 characters, seed `THE FLYING SUBMARINE`, temperature 0.7, pure NumPy:

| Model | Context | ms per character | 400 chars |
|---|---|---|---|
| RNN, hidden state carried | 120 | 0.14 | 0.06 s |
| Transformer matched | 120 | 3.46 | 1.38 s |
| Transformer matched | 256 | 3.58 | 1.43 s |
| Transformer large | 256 | 29.17 | 11.67 s |

The RNN remains ~25x faster per character than the size-matched Transformer, and
~200x faster than the 25M one. That is the O(1) versus O(C) distinction with
real numbers attached.

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

- **The RNN's reported loss (0.45) is a training loss** from the original
  notebook, so it is not directly comparable to the held-out validation losses
  this project reports. Use the Transformer-to-Transformer comparisons for the
  rigorous claims, and treat the RNN figure as indicative.
- **Validation loss is not the same as writing quality.** Perplexity is a proxy;
  the side-by-side output is the thing to actually read in class.
- **The corpus is fixed and small** (5.77M characters, ten novels). Conclusions
  about context length or scale at this size may not hold at larger sizes. That
  caveat is itself a useful teaching point.
- **The large model is slow to demonstrate in NumPy.** If the size story matters
  more than the transparency story, an ONNX Runtime path
  (`export_transformer_onnx.py`) is started but not yet correct; see the status
  note in that file.
