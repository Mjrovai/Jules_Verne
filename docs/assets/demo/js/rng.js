/* ==========================================================================
   Random number generation for the in-browser demo.

   Scope, stated plainly
   ---------------------
   This generator is **not** bit-identical to NumPy's PCG64, so a given seed
   does not reproduce the Python engine's exact characters. Matching it byte for
   byte would mean porting NumPy's `SeedSequence`, `PCG64` and the Walker-alias
   path of `Generator.choice` exactly, and an approximation that is *nearly*
   right is worse than an honest one: it would look interchangeable while
   quietly disagreeing.

   What IS guaranteed, and what the test suite checks:

   * the model arithmetic is identical to the Python engine -- verified by
     greedy decoding against `vernebot/transformer.py` in `tests/`
   * a seed is repeatable within the page
   * the sampler (temperature, top-k, greedy) is the same algorithm

   So the seed means "reproducible in the page", not "reproduces the notebook".
   ========================================================================== */

/**
 * xoshiro128** -- small, fast, good statistical quality, trivially portable.
 * Used instead of Math.random() because a seed must give a repeatable sequence.
 */
class RNG {
  constructor(seed = 0) {
    // SplitMix32 expands the seed: xoshiro is sensitive to a sparse initial
    // state, and nearby seeds would otherwise produce correlated streams.
    let state = (seed >>> 0) || 0x9e3779b9;
    const next = () => {
      state = (state + 0x9e3779b9) >>> 0;
      let z = state;
      z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
      z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
      return (z ^ (z >>> 15)) >>> 0;
    };
    this.s = new Uint32Array([next(), next(), next(), next()]);
  }

  /** Next 32-bit unsigned integer (xoshiro128**). */
  nextUint32() {
    const s = this.s;
    const result = Math.imul(s[1], 5) >>> 0;
    const rotated = ((result << 7) | (result >>> 25)) >>> 0;
    const out = Math.imul(rotated, 9) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = ((s[3] << 11) | (s[3] >>> 21)) >>> 0;
    return out;
  }

  /** Uniform double in [0, 1), built from 53 bits as a double can hold. */
  random() {
    const hi = this.nextUint32() >>> 5;      // 27 bits
    const lo = this.nextUint32() >>> 6;      // 26 bits
    return (hi * 67108864 + lo) / 9007199254740992;
  }
}

/**
 * Sample one index from `probs`, which must already sum to 1.
 *
 * A plain cumulative scan. It is exact (no binning), and consumes exactly one
 * uniform draw, so the sampler stays comparable between models.
 */
function categorical(probs, rng) {
  const draw = rng.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (draw < cumulative) return i;
  }
  return probs.length - 1;
}

export { RNG, categorical };
