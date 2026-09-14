/* ==========================================================================
   The Transformer, ported from vernebot/transformer.py.

   Decoder-only, pre-norm, learned absolute positions, weight-tied output head.
   Two things here mirror decisions in the Python engine and are load-bearing:

   **The KV cache is a ring.** Once generation passes the context length, each
   new character overwrites the oldest cache slot and costs exactly one pass
   through the stack. Replaying the window instead is exact but costs `context`
   passes per character; measured on the Python side that was 122 s versus 1.2 s
   for 367 characters.

   **The slot doubles as the position.** After the ring wraps, positions restart
   at 0, so the window is always "the last C characters, numbered from its own
   start" -- which is what training saw. Using the absolute position here gives
   a generated token the embedding of a much later slot, and the text collapses
   into gibberish from the first wrapped character.

   All weight tensors are (out, in), so a linear layer is always `W @ x`.
   ========================================================================== */

import { Mat } from './loader.js';
import { geluVec, layerNorm, softmax } from './nn.js';
import { RNG, categorical } from './rng.js';

class TransformerModel {
  constructor(info, tensors, vocab) {
    this.info = info;
    this.vocab = vocab;
    this.dModel = info.d_model;
    this.nLayers = info.n_layers;
    this.nHeads = info.n_heads;
    this.dHead = info.d_model / info.n_heads;
    this.dFf = info.d_ff;
    this.context = info.context;

    this.wte = tensors.wte.data;
    this.wpe = tensors.wpe ? tensors.wpe.data : null;
    this.head = new Mat(tensors.head.shape[0], tensors.head.shape[1], tensors.head.data);

    this.layers = [];
    for (let l = 0; l < this.nLayers; l++) {
      const m = (name) => {
        const t = tensors[`${name}_${l}`];
        return new Mat(t.shape[0], t.shape[1], t.data);
      };
      this.layers.push({
        wq: m('wq_w'), wk: m('wk_w'), wv: m('wv_w'), wo: m('wo_w'),
        fc: m('fc_w'), proj: m('proj_w'),
        ln1g: tensors[`ln1_g_${l}`].data, ln1b: tensors[`ln1_b_${l}`].data,
        ln2g: tensors[`ln2_g_${l}`].data, ln2b: tensors[`ln2_b_${l}`].data,
        wqb: tensors[`wq_b_${l}`].data, wkb: tensors[`wk_b_${l}`].data,
        wvb: tensors[`wv_b_${l}`].data, wob: tensors[`wo_b_${l}`].data,
        fcb: tensors[`fc_b_${l}`].data, projb: tensors[`proj_b_${l}`].data,
      });
    }
    this.lnfG = tensors.lnf_g.data;
    this.lnfB = tensors.lnf_b.data;

    this.cache = this.freshCache();
  }

  freshCache() {
    return Array.from({ length: this.nLayers }, () => ({
      k: new Float32Array(this.context * this.dModel),
      v: new Float32Array(this.context * this.dModel),
    }));
  }

  /**
   * Ingest one token at absolute position `pos`, returning logits.
   * Used for the prompt and then, in ring mode, for each generated character.
   */
  step(index, pos) {
    const D = this.dModel;
    const slot = pos % this.context;
    let x = new Float32Array(D);
    const embBase = index * D;
    for (let i = 0; i < D; i++) x[i] = this.wte[embBase + i] + this.wpe[slot * D + i];

    const wrapped = pos >= this.context;
    const visible = wrapped ? this.context - 1 : pos + 1;
    const scale = 1 / Math.sqrt(this.dHead);

    for (let l = 0; l < this.nLayers; l++) {
      const L = this.layers[l];
      const normed = layerNorm(x, L.ln1g, L.ln1b);

      const q = L.wq.times(normed);
      const k = L.wk.times(normed);
      const v = L.wv.times(normed);
      for (let i = 0; i < D; i++) { q[i] += L.wqb[i]; k[i] += L.wkb[i]; v[i] += L.wvb[i]; }

      this.cache[l].k.set(k, slot * D);
      this.cache[l].v.set(v, slot * D);

      const attended = new Float32Array(D);
      for (let h = 0; h < this.nHeads; h++) {
        const off = h * this.dHead;
        const scores = new Float64Array(visible);
        let t = 0;
        for (let p = 0; p < this.context; p++) {
          // Skip the slot just overwritten: it now holds this token, which must
          // not be visible to itself. That leaves context-1 earlier positions.
          if (wrapped && p === slot) continue;
          if (t >= visible) break;
          const kb = p * D + off;
          let s = 0;
          for (let d = 0; d < this.dHead; d++) s += q[off + d] * this.cache[l].k[kb + d];
          scores[t++] = s * scale;
        }
        const weights = softmax(scores);
        for (let d = 0; d < this.dHead; d++) {
          let s = 0;
          let tt = 0;
          for (let p = 0; p < this.context; p++) {
            if (wrapped && p === slot) continue;
            if (tt >= visible) break;
            s += weights[tt++] * this.cache[l].v[p * D + off + d];
          }
          attended[off + d] = s;
        }
      }

      const proj = L.wo.times(attended);
      const res1 = new Float32Array(D);
      for (let i = 0; i < D; i++) res1[i] = x[i] + proj[i] + L.wob[i];

      const normed2 = layerNorm(res1, L.ln2g, L.ln2b);
      // Bias first, then GELU. Adding fc_b after the activation looks almost
      // right and is wrong: it shifts the output by the bias instead of moving
      // the pre-activation, which perturbs the whole MLP branch.
      const pre = L.fc.times(normed2);
      for (let i = 0; i < pre.length; i++) pre[i] += L.fcb[i];
      const hidden = geluVec(pre);
      const mlp = L.proj.times(hidden);

      x = new Float32Array(D);
      for (let i = 0; i < D; i++) x[i] = res1[i] + mlp[i] + L.projb[i];
    }

    return this.head.times(layerNorm(x, this.lnfG, this.lnfB));
  }

  *generate(seedText, options) {
    const {
      numGenerate = 300, temperature = 0.7, topK = null, greedy = false,
      seedValue = 0,
    } = options || {};

    const charToIdx = new Map();
    this.vocab.forEach((c, i) => charToIdx.set(c, i));

    const cleaned = [...seedText].filter((c) => charToIdx.has(c)).join('')
      || this.vocab[0];

    // Reserve one slot so the first generated token never lands on a position a
    // prompt token already used.
    const prompt = [...cleaned].map((c) => charToIdx.get(c)).slice(-(this.context - 1));

    this.cache = this.freshCache();
    let logits = null;
    for (let pos = 0; pos < prompt.length; pos++) logits = this.step(prompt[pos], pos);

    yield { char: cleaned, done: false };

    const rng = new RNG(seedValue);
    let pos = prompt.length;
    for (let i = 0; i < numGenerate; i++) {
      const next = sample(logits, temperature, topK, greedy, rng);
      yield { char: this.vocab[next], done: i === numGenerate - 1 };
      logits = this.step(next, pos);
      pos += 1;
    }
  }
}

/** Shared sampler, identical in behaviour to the GRU's. */
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
    const order = Array.from({ length: n }, (_v, i) => i).sort((a, b) => scaled[b] - scaled[a]);
    const cutoff = scaled[order[topK - 1]];
    for (let i = 0; i < n; i++) if (scaled[i] < cutoff) scaled[i] = -Infinity;
  }

  return categorical(softmax(Array.from(scaled)), rng);
}

export { TransformerModel, sample };
