/* ==========================================================================
   Parity test: the browser inference ports against the Python engines.

   Run after `python tests/dump_python_reference.py`:

       node tests/check_js_parity.mjs

   Compares logits, not generated text. Text is the wrong signal here: the
   browser loads float16 weights, so a near-tied argmax flips and every later
   character changes with it. Logits degrade gracefully -- they differ by
   roughly the quantisation error, which is a number you can put a threshold on.

   Reads files with node:fs rather than fetch, so the same module code runs in
   the browser and here.
   ========================================================================== */

import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { fp16ToFp32 } from '../docs/assets/demo/js/loader.js';
import { GRUModel } from '../docs/assets/demo/js/gru.js';
import { TransformerModel } from '../docs/assets/demo/js/transformer.js';
import { reflow } from '../docs/assets/demo/js/reflow.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const MODELS = join(ROOT, 'docs', 'assets', 'models');
const BOOKS = join(ROOT, 'books_clean');

// Share of words past the window that must appear in the corpus. The rebuilt
// cache scores about 97%; letting the ring wrap instead scores about 50%.
const MIN_REAL_WORDS = 0.85;

// float16 carries ~3 decimal digits, so each dot product is perturbed by about
// 1e-3 relative, and a 5-layer stack over a 20-character prompt accumulates
// that. An absolute tolerance is therefore the wrong instrument: logits span
// tens of units, so the same relative error is a much larger absolute number on
// the tail than near the mean. The thresholds below are relative to the spread
// of the logits, plus a rank check, which is what actually matters -- do the two
// implementations rank the same characters highly?
const MAX_REL_TOL = 0.05;       // 5% of the logit range
const MIN_TOP8_OVERLAP = 7;     // of 8

const reference = JSON.parse(readFileSync(join(HERE, 'reference.json'), 'utf8'));
const manifest = JSON.parse(readFileSync(join(MODELS, 'manifest.json'), 'utf8'));
const vocab = manifest.vocab.split('');

let failures = 0;

function check(name, ok, detail) {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? `  -- ${detail}` : ''}`);
  if (!ok) failures += 1;
}

/** Largest deviation between two logit vectors, as a fraction of the spread. */
function relativeError(got, want) {
  const spread = rangeOf(want);
  if (spread === 0) return 0;
  let maxDiff = 0;
  for (let i = 0; i < want.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(got[i] - want[i]));
  }
  return maxDiff / spread;
}

function rangeOf(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return hi - lo;
}

function loadTensors(key) {
  const info = manifest.models[key];
  if (!info) return null;
  const buf = readFileSync(join(MODELS, info.file));
  const flat = fp16ToFp32(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const tensors = {};
  for (const [name, spec] of Object.entries(info.tensors)) {
    tensors[name] = {
      shape: spec.shape,
      data: flat.subarray(spec.offset, spec.offset + spec.length),
    };
  }
  return { info, tensors };
}

/* ------------------------------------------------------------------- RNN */

{
  console.log('RNN (GRU 1024)');
  const loaded = loadTensors('rnn');
  check('manifest lists the RNN', !!loaded);
  if (loaded) {
    const model = new GRUModel(loaded.info, loaded.tensors, vocab, loaded.info.context);
    model.reset();

    let logits = null;
    for (const ch of reference.prompt) logits = model.step(vocab.indexOf(ch));

    const want = reference.models.rnn.logits;
    check('logit vector length', logits.length === want.length,
      `${logits.length} vs ${want.length}`);

    const rel = relativeError(logits, want);
    check('logits within float16 tolerance (relative)', rel < MAX_REL_TOL,
      `max |diff| = ${rel.toFixed(4)} of a ${(rangeOf(want)).toFixed(1)}-unit spread`);

    // Character-level agreement matters more than raw magnitude: does the port
    // rank the same characters highly?
    const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
    const jsTop = order.slice(0, 8);
    const pyTop = reference.models.rnn.top;
    const overlap = jsTop.filter((i) => pyTop.includes(i)).length;
    check('top-8 characters overlap', overlap >= MIN_TOP8_OVERLAP,
      `js ${jsTop.map((i) => vocab[i]).join('')} vs py ${pyTop.map((i) => vocab[i]).join('')}`);
    check('argmax matches', jsTop[0] === pyTop[0],
      `js '${vocab[jsTop[0]]}' vs py '${vocab[pyTop[0]]}'`);
  }
}

/* ----------------------------------------------------------- Transformer */

if (manifest.models['tx-paired']) {
  console.log('\nTransformer (matched, ctx 256)');
  const loaded = loadTensors('tx-paired');
  const info = loaded.info;

  // Use the module the page actually runs. An earlier version of this test
  // reimplemented the forward pass inline, which meant it kept passing while
  // the real module was broken -- it was checking a copy, not the code.
  const model = new TransformerModel(info, loaded.tensors, vocab);
  const chars = reference.prompt.slice(-(info.context - 1));

  let logits = null;
  for (let pos = 0; pos < chars.length; pos++) {
    logits = model.step(vocab.indexOf(chars[pos]), pos);
  }

  check('prompt length matches the page',
    chars.length === reference.models['tx-paired'].prompt_length,
    `${chars.length} vs ${reference.models['tx-paired'].prompt_length}`);

  const want = reference.models['tx-paired'].logits;
  check('logit vector length', logits.length === want.length,
    `${logits.length} vs ${want.length}`);

  const rel = relativeError(logits, want);
  check('logits within float16 tolerance (relative)', rel < MAX_REL_TOL,
    `max |diff| = ${rel.toFixed(4)} of a ${rangeOf(want).toFixed(1)}-unit spread`);

  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  const jsTop = order.slice(0, 8);
  const pyTop = reference.models['tx-paired'].top;
  const overlap = jsTop.filter((i) => pyTop.includes(i)).length;
  check('top-8 characters overlap', overlap >= MIN_TOP8_OVERLAP,
    `js ${jsTop.map((i) => vocab[i]).join('')} vs py ${pyTop.map((i) => vocab[i]).join('')}`);
  check('argmax matches', jsTop[0] === pyTop[0],
    `js '${vocab[jsTop[0]]}' vs py '${vocab[pyTop[0]]}'`);

  // Generation past the window must stay coherent, and "coherent" has to be
  // measured rather than assumed: the length of the output says nothing. Before
  // the cache was rebuilt periodically, the text past the window degraded into
  // word salad that this check now catches -- real words drop to about half.
  const corpusWords = new Set();
  for (const f of readdirSync(BOOKS)) {
    if (!f.endsWith('.txt')) continue;
    for (const w of readFileSync(join(BOOKS, f), 'utf8').toLowerCase().match(/[a-z]+/g) || []) {
      corpusWords.add(w);
    }
  }

  let out = '';
  for (const { char } of model.generate('THE MOON',
    { numGenerate: info.context + 350, temperature: 0.7, seedValue: 1 })) out += char;
  // Skip the first window: the interesting part is what comes after it.
  const past = (out.slice(info.context).toLowerCase().match(/[a-z]+/g) || []);
  const known = past.filter((w) => corpusWords.has(w)).length;
  const ratio = past.length ? known / past.length : 0;
  check('generation past the context length stays coherent', ratio >= MIN_REAL_WORDS,
    `${(ratio * 100).toFixed(1)}% real words in the ${past.length} written past the window`
    + ` (threshold ${(MIN_REAL_WORDS * 100).toFixed(0)}%)`);
  check('generation contains no NaN artifacts',
    !/NaN|undefined/.test(out));
} else {
  console.log('\nTransformer: skipped (not in the manifest)');
}

/* ------------------------------------------------------------------ reflow */

{
  console.log('\nText reflow (JS vs Python)');
  const cases = JSON.parse(readFileSync(join(HERE, 'reflow_cases.json'), 'utf8')).cases;
  let mismatches = 0;
  for (const { input, expected } of cases) {
    if (reflow(input) !== expected) {
      mismatches += 1;
      console.log(`    mismatch for ${JSON.stringify(input)}`);
      console.log(`      js ${JSON.stringify(reflow(input))}`);
      console.log(`      py ${JSON.stringify(expected)}`);
    }
  }
  check(`${cases.length} reflow cases match Python`, mismatches === 0,
    mismatches ? `${mismatches} differ` : 'identical output');
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
