/* ==========================================================================
   Jules Verne Bot — demo page controller.

   Loading strategy
   ----------------
   The RNN is 8 MB of 16-bit weights and the Transformer is another 8 MB. Both
   could be fetched together, but then the page is dead for however long 16 MB
   takes. Instead:

     1. the small manifest is read and the RNN is fetched, with a progress bar
     2. as soon as it is ready the page becomes usable
     3. the Transformer is fetched in the background and its own bar fills in
     4. once it lands, the model selector and the comparison unlock

   Everything runs locally after that: no network per generation, no server.
   ========================================================================== */

import { loadBundle } from './js/loader.js';
import { GRUModel, sample as gruSample } from './js/gru.js';
import { TransformerModel } from './js/transformer.js';
import { reflow } from './js/reflow.js';
import { RNG } from './js/rng.js';

const BASE = 'assets/models';

const SEEDS = [
  'THE FLYING SUBMARINE', 'CAPTAIN NEMO', 'THE MOON',
  'MY UNCLE', 'At the bottom of the sea',
];

const el = (id) => document.getElementById(id);
const ui = {
  model: el('model'), seed: el('seed'), chips: el('chips'),
  generate: el('generate'), compare: el('compare'),
  temperature: el('temperature'), temperatureReadout: el('temperature-readout'),
  length: el('length'), lengthReadout: el('length-readout'),
  greedy: el('greedy'), reflowToggle: el('reflow'), seedValue: el('seed-value'),
  story: el('story'), stats: el('stats'), copy: el('copy'),
  compareGrid: el('compare-grid'), compareStats: el('compare-stats'),
  errorSlot: el('error-slot'),
  loaderRnn: el('loader-rnn'), rnnBar: el('rnn-bar'), rnnPct: el('rnn-pct'), rnnLabel: el('rnn-label'),
  loaderTx: el('loader-tx'), txBar: el('tx-bar'), txPct: el('tx-pct'), txLabel: el('tx-label'),
};

/** Loaded models, keyed as in the manifest. */
const models = {};
let vocab = null;
let busy = false;
let lastText = '';
let typewriter = null;

/* ------------------------------------------------------------------ helpers */

function showError(message) {
  ui.errorSlot.innerHTML = '';
  if (!message) return;
  const div = document.createElement('div');
  div.className = 'alert';
  div.setAttribute('role', 'alert');
  div.textContent = message;
  ui.errorSlot.appendChild(div);
}

function setBusy(state) {
  busy = state;
  ui.generate.disabled = state || !models.rnn;
  const bothReady = models.rnn && models.transformer;
  ui.compare.disabled = state || !bothReady;
  ui.generate.textContent = state ? 'Writing…' : 'Generate';
}

function markLoader(node, bar, pct, state, text) {
  node.classList.toggle('ready', state === 'ready');
  node.classList.toggle('failed', state === 'failed');
  if (bar && state === 'ready') bar.style.width = '100%';
  if (pct) pct.textContent = text;
}

/** Show download progress as a percentage, or as MB when the size is unknown. */
function progressText(received, total) {
  const mb = (n) => (n / 1048576).toFixed(1);
  if (!total) return `${mb(received)} MB`;
  return `${Math.round((received / total) * 100)}% · ${mb(received)}/${mb(total)} MB`;
}

/* -------------------------------------------------------------- generation */

/**
 * Reveal text character by character so the page feels alive while it writes.
 *
 * The animation follows the writing by scrolling to the bottom, but it must not
 * fight the reader: if you scroll up to re-read something while the text is
 * still arriving, the next frame would drag you back down again. So the
 * follow-the-bottom behaviour switches itself off for the rest of the run as
 * soon as a scroll arrives that the animation did not cause.
 */
let followOutput = true;

function atBottom(node, slack = 4) {
  return node.scrollHeight - node.scrollTop - node.clientHeight <= slack;
}

function reveal(text, seedLength, animate = true) {
  if (typewriter) cancelAnimationFrame(typewriter);
  ui.story.innerHTML = '';

  const seedSpan = document.createElement('span');
  seedSpan.className = 'seed';
  seedSpan.textContent = text.slice(0, seedLength);
  const body = document.createElement('span');
  ui.story.append(seedSpan, body);

  const rest = text.slice(seedLength);
  if (!animate || !rest.length) {
    body.textContent = rest;
    return;
  }

  const caret = document.createElement('span');
  caret.className = 'caret';
  caret.textContent = '|';
  ui.story.append(caret);

  followOutput = true;
  let programmatic = false;

  // A scroll event with the flag clear means the reader did it, so stop
  // following. Wheel and touch are also listened for directly, because on some
  // trackpads the scroll event arrives a frame late.
  const reading = () => { followOutput = false; };
  const onScroll = () => {
    if (programmatic) { programmatic = false; return; }
    if (!atBottom(ui.story)) reading();
  };
  ui.story.addEventListener('scroll', onScroll, { passive: true });
  ui.story.addEventListener('wheel', reading, { passive: true });
  ui.story.addEventListener('touchmove', reading, { passive: true });

  const step = Math.max(3, Math.ceil(rest.length / 40));
  let shown = 0;
  const tick = () => {
    shown = Math.min(rest.length, shown + step);
    body.textContent = rest.slice(0, shown);
    if (followOutput) {
      programmatic = true;
      ui.story.scrollTop = ui.story.scrollHeight;
    }
    if (shown < rest.length) {
      typewriter = requestAnimationFrame(tick);
    } else {
      caret.remove();
      typewriter = null;
      ui.story.removeEventListener('scroll', onScroll);
      ui.story.removeEventListener('wheel', reading);
      ui.story.removeEventListener('touchmove', reading);
    }
  };
  typewriter = requestAnimationFrame(tick);
}

function runWith(modelKey, seedText, options) {
  const model = models[modelKey];
  const cleaned = [...seedText].filter((c) => model.instance.vocab
    ? model.instance.vocab.includes(c)
    : true).join('') || seedText;

  const iterator = model.instance.generate(cleaned, options);
  const started = performance.now();

  return new Promise((resolve) => {
    let out = '';
    const target = cleaned.length + options.numGenerate;
    const pump = () => {
      const budget = performance.now() + 28;
      for (;;) {
        const { value, done } = iterator.next();
        if (done) break;
        out += value.char;
        if (performance.now() > budget) break;
      }
      if ([...out].length >= target) {
        resolve({
          text: options.reflow ? reflow(out) : out,
          seedLength: [...cleaned].length,
          elapsed: performance.now() - started,
        });
      } else {
        setTimeout(pump, 0);
      }
    };
    pump();
  });
}

async function generate() {
  if (busy || !models.rnn) return;
  const seedText = ui.seed.value.trim();
  if (!seedText) { showError('Enter a seed word or phrase first.'); return; }

  showError('');
  setBusy(true);
  ui.stats.textContent = '';
  followOutput = true;

  const key = ui.model.value || 'rnn';
  const options = {
    numGenerate: Number(ui.length.value),
    temperature: Number(ui.temperature.value),
    greedy: ui.greedy.checked,
    seedValue: ui.seedValue.value.trim() === '' ? Date.now() % 100000 : Number(ui.seedValue.value),
    reflow: ui.reflowToggle.checked,
  };

  try {
    const result = await runWith(key, seedText, options);
    lastText = result.text;
    ui.copy.disabled = false;
    reveal(result.text, result.seedLength);
    const label = models[key].label;
    const mode = options.greedy ? 'greedy' : `T=${options.temperature.toFixed(2)}`;
    ui.stats.textContent =
      `${result.text.length} chars · ${mode} · ${label} · ${(result.elapsed / 1000).toFixed(2)} s`;
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

/* --------------------------------------------------------------- comparison */

async function runComparison() {
  if (busy || !models.rnn || !models.transformer) return;
  const seedText = ui.seed.value.trim();
  if (!seedText) { showError('Enter a seed word or phrase first.'); return; }

  showError('');
  setBusy(true);
  ui.compareStats.textContent = 'running…';

  const options = {
    numGenerate: Number(ui.length.value),
    temperature: Number(ui.temperature.value),
    greedy: ui.greedy.checked,
    seedValue: ui.seedValue.value.trim() === '' ? 7 : Number(ui.seedValue.value),
  };

  const order = [
    { key: 'rnn', name: 'RNN · GRU 1024', note: '4.10M · ctx 120' },
    { key: 'transformer', name: 'Transformer · matched', note: '4.05M · ctx 256' },
  ];

  ui.compareGrid.innerHTML = '';
  const cards = {};
  for (const item of order) {
    const card = document.createElement('div');
    card.className = `card pending ${item.key}`;
    card.innerHTML =
      `<header><span>${item.name}</span><span class="meta">${item.note}</span></header>` +
      `<div class="body"><span class="placeholder">Writing…</span></div>`;
    ui.compareGrid.appendChild(card);
    cards[item.key] = card;
  }

  const started = performance.now();
  // Sequential rather than parallel: both engines are CPU-bound and would
  // otherwise compete for the same single thread.
  for (const item of order) {
    const result = await runWith(item.key, seedText, options);
    const body = cards[item.key].querySelector('.body');
    body.innerHTML = '';
    const span = document.createElement('span');
    span.className = 'seed';
    span.textContent = result.text.slice(0, result.seedLength);
    body.append(span, document.createTextNode(result.text.slice(result.seedLength)));
    cards[item.key].classList.remove('pending');
    cards[item.key].querySelector('.meta').textContent =
      `${item.note} · ${(result.elapsed / 1000).toFixed(2)} s`;
  }

  ui.compareStats.textContent =
    `${((performance.now() - started) / 1000).toFixed(1)} s total · reflow ${ui.reflowToggle.checked ? 'on' : 'off'}`;
  setBusy(false);
}

/* ----------------------------------------------------------------- controls */

function syncReadouts() {
  ui.temperatureReadout.textContent = Number(ui.temperature.value).toFixed(2);
  ui.lengthReadout.textContent = ui.length.value;
}

function buildChips() {
  for (const seed of SEEDS) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip';
    button.textContent = seed;
    button.addEventListener('click', () => { ui.seed.value = seed; ui.seed.focus(); });
    ui.chips.appendChild(button);
  }
}

function refreshModelSelect() {
  const ready = [
    models.rnn && { key: 'rnn', label: 'RNN · GRU 1024 (4.10M, ctx 120)' },
    models.transformer && { key: 'transformer', label: 'Transformer · matched (4.05M, ctx 256)' },
  ].filter(Boolean);

  if (!ready.length) return;
  const previous = ui.model.value;
  ui.model.innerHTML = '';
  for (const item of ready) {
    const option = document.createElement('option');
    option.value = item.key;
    option.textContent = item.label;
    ui.model.appendChild(option);
  }
  ui.model.value = ready.some((r) => r.key === previous) ? previous : ready[0].key;
  ui.model.disabled = ready.length < 2;
}

/* -------------------------------------------------------------------- boot */

async function boot() {
  buildChips();
  syncReadouts();

  ui.temperature.addEventListener('input', syncReadouts);
  ui.length.addEventListener('input', syncReadouts);
  ui.generate.addEventListener('click', generate);
  ui.compare.addEventListener('click', runComparison);
  ui.model.addEventListener('change', () => { if (lastText) generate(); });
  ui.seed.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); generate(); }
  });

  // The output panel scrolls, so it has to be reachable and operable from the
  // keyboard: arrow keys and Page Up/Down once it has focus.
  for (const node of [ui.story, ...document.querySelectorAll('.card .body')]) {
    node.tabIndex = 0;
    node.setAttribute('role', 'region');
    node.setAttribute('aria-label', 'Generated text, scrollable');
  }
  ui.story.tabIndex = 0;
  ui.copy.addEventListener('click', async () => {
    if (!lastText) return;
    try {
      await navigator.clipboard.writeText(lastText);
      ui.copy.textContent = 'Copied';
      setTimeout(() => { ui.copy.textContent = 'Copy'; }, 1400);
    } catch { showError('The browser blocked clipboard access.'); }
  });

  // The manifest carries the vocabulary and the tensor layout for both models.
  let manifest;
  try {
    manifest = await (await fetch(`${BASE}/manifest.json`)).json();
  } catch (error) {
    showError(`Could not load the model manifest: ${error.message}`);
    return;
  }
  vocab = manifest.vocab.split('');

  const makeInstance = (key, bundle) => (bundle.info.architecture === 'rnn'
    ? new GRUModel(bundle.info, bundle.tensors, vocab, bundle.info.context)
    : new TransformerModel(bundle.info, bundle.tensors, vocab));

  const register = (key, bundle, label) => {
    models[key] = {
      instance: makeInstance(key, bundle),
      label,
      kind: bundle.info.architecture,
    };
  };

  // 1. the RNN, which makes the page usable
  try {
    const { models: rnnModels } = await loadBundle(BASE, ['rnn'], (key, received, total) => {
      ui.rnnBar.style.width = total ? `${(received / total) * 100}%` : '100%';
      ui.rnnPct.textContent = progressText(received, total);
    });
    register('rnn', rnnModels.rnn, 'RNN');
    markLoader(ui.loaderRnn, ui.rnnBar, ui.rnnPct, 'ready', 'ready');
    ui.rnnLabel.textContent = 'RNN weights';
    refreshModelSelect();
    setBusy(false);
    ui.story.innerHTML =
      '<span class="placeholder">The RNN is ready. Enter a word and set the machine ' +
      'running — the Transformer is still downloading in the background.</span>';
  } catch (error) {
    markLoader(ui.loaderRnn, ui.rnnBar, ui.rnnPct, 'failed', 'failed');
    showError(`Could not load the RNN weights: ${error.message}`);
    return;
  }

  // 2. the Transformer, in the background
  ui.txPct.textContent = 'starting…';
  try {
    const { models: txModels } = await loadBundle(BASE, ['tx-paired'], (key, received, total) => {
      ui.txBar.style.width = total ? `${(received / total) * 100}%` : '100%';
      ui.txPct.textContent = progressText(received, total);
    });
    register('transformer', txModels['tx-paired'], 'Transformer');
    markLoader(ui.loaderTx, ui.txBar, ui.txPct, 'ready', 'ready');
    ui.txLabel.textContent = 'Transformer weights';
    refreshModelSelect();
    setBusy(false);
    // Replace the "still downloading" note, which is now out of date.
    ui.story.innerHTML =
      '<span class="placeholder">Enter a word and set the machine running, or '
      + 'press <em>Run both</em> in Side by Side to see them on the same seed.</span>';

  } catch (error) {
    markLoader(ui.loaderTx, ui.txBar, ui.txPct, 'failed', 'failed');
    ui.txLabel.textContent = 'Transformer weights (unavailable)';
    showError(`The Transformer could not be loaded: ${error.message}. ` +
      'The RNN still works.');
    ui.story.innerHTML =
      '<span class="placeholder">The RNN is ready. Enter a word and set the machine ' +
      'running.</span>';
  }

  // A small handle for automated checks and console experiments.
  window.vernebot = { models, generate, runComparison, reflow };
}

boot();
