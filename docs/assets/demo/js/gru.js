/* ==========================================================================
   The GRU model, ported from vernebot/engine.py.

   This must reproduce the Python engine exactly, and the gate arithmetic below
   is the part of this whole project that is easiest to get subtly wrong. Keras
   with `reset_after=True` computes:

       z  = sigmoid(x @ Wz + h @ Rz + bz + rbz)
       r  = sigmoid(x @ Wr + h @ Rr + br + rbr)
       hh = tanh(x @ Wh + bh + r * (h @ Rh + rbh))
       h' = z * h + (1 - z) * hh

   Two details that produce plausible-looking nonsense if missed:

   * the reset gate multiplies the *result* of the recurrent matmul, not the
     previous state going into it (`(r * h) @ Rh` is the other common form and
     is wrong here)
   * the recurrent bias `rb*` applies to all three gates, not only the candidate
   ========================================================================== */

import { Mat } from './loader.js';
import { RNG, categorical } from './rng.js';

function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

class GRUModel {
  constructor(info, tensors, vocab, context = 120) {
    this.info = info;
    this.vocab = vocab;
    this.context = context;
    this.dModel = info.d_model;
    this.units = info.units;

    const m = (name) => {
      const t = tensors[name];
      return new Mat(t.shape[0], t.shape[1], t.data);
    };

    this.wte = tensors.wte;                  // (vocab, d_model), row lookup
    this.kernelZ = m('kernel_z');
    this.kernelR = m('kernel_r');
    this.kernelH = m('kernel_h');
    this.recurrentZ = m('recurrent_z');
    this.recurrentR = m('recurrent_r');
    this.recurrentH = m('recurrent_h');
    this.biasInZ = tensors.input_bias_z.data;
    this.biasInR = tensors.input_bias_r.data;
    this.biasInH = tensors.input_bias_h.data;
    this.biasRecZ = tensors.recurrent_bias_z.data;
    this.biasRecR = tensors.recurrent_bias_r.data;
    this.biasRecH = tensors.recurrent_bias_h.data;
    this.denseW = m('dense_w');
    this.denseB = tensors.dense_b.data;

    this.h = new Float32Array(this.units);
    this.z = new Float32Array(this.units);
    this.r = new Float32Array(this.units);
    this.hh = new Float32Array(this.units);
    this.tmpZ = new Float32Array(this.units);
    this.tmpR = new Float32Array(this.units);
    this.tmpH = new Float32Array(this.units);
    this.x = new Float32Array(this.dModel);
  }

  reset() {
    this.h.fill(0);
  }

  /** One timestep. `index` is a character id; returns logits (length vocab). */
  step(index, logits) {
    const x = this.x;
    const base = index * this.dModel;
    for (let i = 0; i < this.dModel; i++) x[i] = this.wte.data[base + i];

    // Every weight tensor is exported as (out, in), so a linear layer is
    // always `W @ x`: `times` takes a vector of the matrix's column length and
    // returns one of its row count.
    const zin = this.kernelZ.times(x);
    const rin = this.kernelR.times(x);
    const hin = this.kernelH.times(x);
    const zrec = this.recurrentZ.times(this.h);
    const rrec = this.recurrentR.times(this.h);
    const hrec = this.recurrentH.times(this.h);

    const h = this.h;
    for (let j = 0; j < this.units; j++) {
      const z = sigmoid(zin[j] + zrec[j] + this.biasInZ[j] + this.biasRecZ[j]);
      const r = sigmoid(rin[j] + rrec[j] + this.biasInR[j] + this.biasRecR[j]);
      const candidate = Math.tanh(hin[j] + this.biasInH[j]
        + r * (hrec[j] + this.biasRecH[j]));
      h[j] = z * h[j] + (1 - z) * candidate;
    }

    // The dense kernel is (vocab, units) under the same contract, so this is
    // also `W @ x`: hidden state of length `units` -> `vocab` logits.
    const projected = this.denseW.times(h);
    const out = logits || new Float32Array(this.vocab.length);
    for (let v = 0; v < out.length; v++) out[v] = projected[v] + this.denseB[v];
    return out;
  }

  /**
   * Generation. Yields {char, done} so the page can reveal text as it is made
   * instead of freezing for a second and then dumping a paragraph.
   */
  *generate(seedText, options) {
    const {
      numGenerate = 300, temperature = 0.7, topK = null, greedy = false,
      seedValue = 0,
    } = options || {};

    const charToIdx = new Map();
    this.vocab.forEach((c, i) => charToIdx.set(c, i));

    const cleaned = [...seedText].filter((c) => charToIdx.has(c)).join('')
      || this.vocab[0];
    const window = [...cleaned].map((c) => charToIdx.get(c)).slice(-this.context);

    this.reset();
    let logits = null;
    for (const idx of window) logits = this.step(idx);

    yield { char: cleaned, done: false };

    const rng = new RNG(seedValue);
    let last = window[window.length - 1];
    for (let i = 0; i < numGenerate; i++) {
      const next = sample(logits, temperature, topK, greedy, rng);
      const ch = this.vocab[next];
      logits = this.step(next);
      last = next;
      yield { char: ch, done: i === numGenerate - 1 };
    }
  }
}

/** Shared sampler: temperature, then top-k, then softmax, then one draw. */
function sample(logits, temperature, topK, greedy, rng) {
  const n = logits.length;
  if (greedy) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }

  const t = Math.max(temperature, 1e-4);
  const scaled = new Float64Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / t;

  if (topK && topK > 0 && topK < n) {
    const order = Array.from({ length: n }, (_v, i) => i)
      .sort((a, b) => scaled[b] - scaled[a]);
    const cutoff = scaled[order[topK - 1]];
    for (let i = 0; i < n; i++) if (scaled[i] < cutoff) scaled[i] = -Infinity;
  }

  let max = -Infinity;
  for (let i = 0; i < n; i++) if (scaled[i] > max) max = scaled[i];
  let total = 0;
  const probs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const e = Math.exp(scaled[i] - max);
    probs[i] = e;
    total += e;
  }
  for (let i = 0; i < n; i++) probs[i] /= total;
  return categorical(probs, rng);
}

export { GRUModel, sample };
