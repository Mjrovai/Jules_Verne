/* ==========================================================================
   Small numeric helpers shared by both models.

   GELU is the one that matters. There are two common implementations -- the
   exact erf-based one that PyTorch uses by default, and a tanh approximation --
   and they are *not* interchangeable in a deep stack:

     tanh approximation   max abs error 4.7e-4, max relative error 22%
     erf (this file)      max abs error 2.1e-7

   A 0.05% error per activation sounds harmless, but it is systematic rather
   than random, so five layers of residual connections compound it into a
   several-percent drift in the logits. Measured here, switching this port from
   the tanh form to erf cut the logit error against the Python engine by more
   than an order of magnitude.

   The Python engine keeps both (`_gelu` and `_gelu_exact`) for the same reason,
   and the parity test pins which one is expected.
   ========================================================================== */

/** Abramowitz & Stegun 7.1.26, max absolute error 1.5e-7. */
export function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return sign * y;
}

/** Exact (erf-based) GELU, matching torch.nn.functional.gelu's default. */
export function gelu(x) {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

/** GELU over a vector, in place-safe (returns a new array). */
export function geluVec(v) {
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    out[i] = 0.5 * x * (1 + erf(x / Math.SQRT2));
  }
  return out;
}

/** Layer normalisation with NumPy's default epsilon. */
export function layerNorm(x, g, b, eps = 1e-5) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;

  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= n;

  const inv = 1 / Math.sqrt(variance + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
  return out;
}

/** Numerically stable logistic sigmoid. */
export function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

/** Softmax over a plain array, returning Float64Array. */
export function softmax(scores) {
  let max = -Infinity;
  for (const s of scores) if (s > max) max = s;
  let total = 0;
  const out = new Float64Array(scores.length);
  for (let i = 0; i < scores.length; i++) {
    const e = Math.exp(scores[i] - max);
    out[i] = e;
    total += e;
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}
