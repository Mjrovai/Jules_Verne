/* ==========================================================================
   Loads the exported float16 weights and presents them as typed arrays.

   The format is a manifest (JSON) plus one flat little-endian float16 binary
   per model. A tensor is a shape plus an offset into that blob, so "loading a
   model" is one fetch and a set of views -- there is no parsing step and no
   HDF5 reader in the browser.

   float16 -> float32 happens here, once, at load. The GPU could keep fp16 but
   JavaScript has no native half type, and doing the widening once is far
   cheaper than doing it inside every matvec.
   ========================================================================== */

/** Widen a little-endian float16 buffer to Float32Array. */
function fp16ToFp32(buffer) {
  const n = Math.floor(buffer.byteLength / 2);
  const out = new Float32Array(n);
  const view = new DataView(buffer);
  for (let i = 0; i < n; i++) {
    const h = view.getUint16(i * 2, true);
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x03ff;

    if (exponent === 0) {
      // subnormal (and zero)
      out[i] = sign * fraction * 5.960464477539063e-8;
    } else if (exponent === 0x1f) {
      out[i] = fraction ? NaN : sign * Infinity;
    } else {
      out[i] = sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }
  return out;
}

/**
 * A dense weight matrix, always stored row-major as **(out, in)**.
 *
 * That single convention is what makes the two models interchangeable here: a
 * linear layer is always `y = W @ x`, whatever the layer is called. Keras and
 * PyTorch store some of these the other way round, so `export_web_models.py`
 * transposes them once at export time instead of the browser doing it on every
 * generated character.
 *
 * A wrong orientation is silent rather than loud -- a transposed matrix still
 * holds plausible numbers, and the error surfaces only as text that is subtly
 * or completely wrong. `times()` therefore checks the input length and throws.
 */
class Mat {
  constructor(rows, cols, data) {
    this.rows = rows;      // out features
    this.cols = cols;      // in features
    this.data = data;
  }

  /**
   * y = W @ x, for x of length `cols`; returns length `rows`.
   *
   * Rows are contiguous in `data`, so the inner loop walks memory sequentially,
   * which is the fast order for a matrix-vector product.
   */
  times(x) {
    if (x.length !== this.cols) {
      throw new Error(
        `Mat.times: expected a vector of length ${this.cols}, got ${x.length}`);
    }
    const { rows, cols, data } = this;
    const out = new Float32Array(rows);
    for (let j = 0; j < rows; j++) {
      const base = j * cols;
      let sum = 0;
      for (let i = 0; i < cols; i++) sum += data[base + i] * x[i];
      out[j] = sum;
    }
    return out;
  }

  /** y = x @ W, for x of length `rows`; returns length `cols`. */
  timesTransposed(x) {
    if (x.length !== this.rows) {
      throw new Error(
        `Mat.timesTransposed: expected a vector of length ${this.rows}, got ${x.length}`);
    }
    const { rows, cols, data } = this;
    const out = new Float32Array(cols);
    for (let r = 0; r < rows; r++) {
      const xr = x[r];
      if (xr === 0) continue;
      const base = r * cols;
      for (let c = 0; c < cols; c++) out[c] += xr * data[base + c];
    }
    return out;
  }
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fetch ${url}: HTTP ${response.status}`);
  return response.json();
}

/**
 * Fetch a binary blob, reporting progress as it arrives.
 *
 * `response.arrayBuffer()` resolves only when the whole body has landed, which
 * gives no way to show a percentage for an 8 MB download. Reading the stream
 * directly lets the page draw a real progress bar instead of a spinner.
 *
 * Falls back to a single read when the body is not a stream (older browsers, or
 * a file:// response), in which case `onProgress` is simply never called.
 */
async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fetch ${url}: HTTP ${response.status}`);

  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body || !response.body.getReader) {
    const buffer = await response.arrayBuffer();
    if (onProgress) onProgress(buffer.byteLength, buffer.byteLength);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received, total);
  }

  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer;
}

/**
 * Load the manifest and one or more weight blobs.
 *
 * `onProgress(key, received, total)` is called as bytes arrive, so the page can
 * show per-model progress. Models load in the order given, so the RNN can be
 * made available before the Transformer starts downloading.
 */
async function loadBundle(baseUrl, wanted, onProgress, version = '') {
  // `version` is appended to every request. The manifest and the weight blobs
  // are served with a long cache lifetime, so without it a returning visitor
  // keeps the previous set -- which is how a page that had been updated to
  // offer three models went on fetching a two-model manifest.
  const bust = version ? `?v=${encodeURIComponent(version)}` : '';
  const manifest = await fetchJSON(`${baseUrl}/manifest.json${bust}`);
  const vocab = manifest.vocab.split('');
  const models = {};

  for (const key of wanted) {
    const info = manifest.models[key];
    if (!info) {
      // Silently skipping this used to surface later as "cannot read
      // properties of undefined", which says nothing about the cause.
      throw new Error(`the manifest has no model called "${key}" `
        + `(it lists: ${Object.keys(manifest.models).join(', ') || 'none'})`);
    }

    const buffer = await fetchWithProgress(
      `${baseUrl}/${info.file}${bust}`,
      onProgress ? (received, total) => onProgress(key, received, total) : null,
    );
    const flat = fp16ToFp32(buffer);

    const tensors = {};
    for (const [name, spec] of Object.entries(info.tensors)) {
      const slice = flat.subarray(spec.offset, spec.offset + spec.length);
      tensors[name] = { shape: spec.shape, data: slice };
    }
    models[key] = { info, tensors };
  }

  return { vocab, manifest, models };
}

export { loadBundle, fetchWithProgress, fp16ToFp32, Mat };
