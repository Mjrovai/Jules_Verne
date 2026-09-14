/* ==========================================================================
   Jules Verne Bot — front-end controller.
   Plain ES2020, no build step, no external dependencies.
   ========================================================================== */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    form: $("form"),
    seed: $("seed"),
    temperature: $("temperature"),
    temperatureReadout: $("temperature-readout"),
    numGenerate: $("num_generate"),
    lengthReadout: $("length-readout"),
    greedy: $("greedy"),
    topK: $("top_k"),
    topKReadout: $("top-k-readout"),
    seedValue: $("seed_value"),
    submit: $("submit"),
    story: $("story"),
    stats: $("stats"),
    copy: $("copy"),
    compare: $("compare"),
    compareGrid: $("compare-grid"),
    compareStats: $("compare-stats"),
    errorSlot: $("error-slot"),
    advanced: $("advanced"),
    reflow: $("reflow"),
    model: $("model"),
    modelField: $("model-field"),
    modelBlurb: $("model-blurb"),
    compareModels: $("compare-models"),
    modelGrid: $("model-grid"),
  };

  const state = {
    busy: false,
    lastText: "",
    typewriter: null,
    maxGenerate: 3000,
  };

  const PLACEHOLDER_HTML =
    '<span class="placeholder">The page is idle. Enter a word on the left — try ' +
    '“THE FLYING SUBMARINE” — and set the machine running.</span>';

  /**
   * One-click presets. They double as bookmarkable deep links, e.g.
   *   /?seed=CAPTAIN%20NEMO&temperature=0.5&length=1200&run=1
   */
  const PRESETS = [
    { label: "Default",     seed: "THE FLYING SUBMARINE",    temperature: 0.7, length: 900 },
    { label: "Cautious",    seed: "CAPTAIN NEMO",            temperature: 0.5, length: 900 },
    { label: "Adventurous", seed: "Twenty Thousand Leagues", temperature: 1.0, length: 900 },
    { label: "The Moon",    seed: "THE MOON",                temperature: 0.7, length: 900 },
    { label: "Wild",        seed: "MY UNCLE",                temperature: 1.2, length: 900 },
  ];

  /* ------------------------------------------------------------ helpers */

  function showError(message) {
    els.errorSlot.innerHTML = "";
    if (!message) return;
    const div = document.createElement("div");
    div.className = "alert";
    div.setAttribute("role", "alert");
    div.textContent = message;
    els.errorSlot.appendChild(div);
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let detail = `Request failed (HTTP ${response.status})`;
      try {
        const body = await response.json();
        if (body && body.detail) detail = typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail);
      } catch (_) { /* non-JSON error body */ }
      throw new Error(detail);
    }
    return response.json();
  }

  function buildPayload(seedText, temperature, numGenerate) {
    const topK = Number(els.topK.value);
    const seedRaw = els.seedValue.value.trim();
    return {
      seed: seedText,
      model: els.model && els.model.value ? els.model.value : null,
      temperature: Number(temperature.toFixed(2)),
      num_generate: Math.round(numGenerate),
      top_k: topK > 0 ? topK : null,
      greedy: els.greedy.checked,
      seed_value: seedRaw === "" ? null : Number(seedRaw),
      reflow: els.reflow.checked,
    };
  }

  /**
   * Print `text` into `target`, colouring the first `seedLength` characters
   * as the seed and revealing the rest gradually.
   */
  function reveal(target, text, seedLength, { animate = true } = {}) {
    if (state.typewriter) cancelAnimationFrame(state.typewriter);
    target.innerHTML = "";

    const seedSpan = document.createElement("span");
    seedSpan.className = "seed";
    seedSpan.textContent = text.slice(0, seedLength);

    const rest = text.slice(seedLength);
    const body = document.createElement("span");

    const caret = document.createElement("span");
    caret.className = "caret";
    caret.textContent = "|";

    target.append(seedSpan, body);
    if (animate && rest.length) target.append(caret);
    else body.textContent = rest;

    if (!animate || !rest.length) return;

    // Reveal over at most ~900 ms regardless of length, so long generations
    // still feel immediate. Chunked to keep the frame budget small.
    const CHUNK = Math.max(4, Math.ceil(rest.length / 45));
    let shown = 0;
    const step = () => {
      shown = Math.min(rest.length, shown + CHUNK);
      body.textContent = rest.slice(0, shown);
      target.scrollTop = target.scrollHeight;
      if (shown < rest.length) {
        state.typewriter = requestAnimationFrame(step);
      } else {
        caret.remove();
        state.typewriter = null;
      }
    };
    state.typewriter = requestAnimationFrame(step);
  }

  function setBusy(busy, label) {
    state.busy = busy;
    els.submit.disabled = busy;
    els.compare.disabled = busy;
    els.submit.textContent = busy ? (label || "Writing…") : "Generate";
  }

  /* --------------------------------------------------------- live labels */

  els.temperature.addEventListener("input", () => {
    els.temperatureReadout.textContent = Number(els.temperature.value).toFixed(2);
  });

  els.numGenerate.addEventListener("input", () => {
    els.lengthReadout.textContent = els.numGenerate.value;
  });

  els.topK.addEventListener("input", () => {
    const value = Number(els.topK.value);
    els.topKReadout.textContent = value > 0 ? `k = ${value}` : "off";
  });

  document.querySelectorAll(".chip[data-seed]").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.seed.value = chip.dataset.seed;
      els.seed.focus();
    });
  });

  els.greedy.addEventListener("change", () => {
    const on = els.greedy.checked;
    els.temperature.disabled = on;
    els.topK.disabled = on;
    els.submit.textContent = "Set the machine running";
  });

  /* ------------------------------------------------------------- generate */

  /** Read the controls, call the API, and render the result. */
  async function runGeneration() {
    if (state.busy) return;

    const seedText = els.seed.value.trim();
    if (!seedText) {
      showError("Enter a seed word or phrase first.");
      return;
    }

    showError("");
    setBusy(true, "Writing…");
    els.stats.textContent = "";

    try {
      const payload = buildPayload(seedText, Number(els.temperature.value),
                                   Number(els.numGenerate.value));
      const result = await api("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      state.lastText = result.text;
      els.copy.disabled = false;

      const mode = result.greedy
        ? "greedy"
        : `T=${result.temperature.toFixed(2)}` +
          (result.top_k ? ` · top-k ${result.top_k}` : "") +
          (result.seed_value !== null ? ` · seed ${result.seed_value}` : "");

      reveal(els.story, result.text, result.seed.length);
      els.stats.textContent =
        `${result.text.length} chars · ${mode} · ${result.elapsed_ms} ms`;
    } catch (error) {
      showError(error.message);
      if (!state.lastText) els.story.innerHTML = PLACEHOLDER_HTML;
    } finally {
      setBusy(false);
    }
  }

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    runGeneration();
  });

  /* ------------------------------------------------------------ compare */

  els.compare.addEventListener("click", async () => {
    if (state.busy) return;

    const seedText = els.seed.value.trim();
    if (!seedText) {
      showError("Enter a seed word or phrase first.");
      return;
    }

    showError("");
    setBusy(true, "Writing…");
    els.compareStats.textContent = "running…";

    const temps = [0.5, 0.7, 1.0];
    const cards = [...els.compareGrid.querySelectorAll(".compare-card")];

    cards.forEach((card) => {
      card.classList.add("loading");
      card.querySelector(".body").innerHTML =
        '<span class="placeholder">Writing…</span>';
    });

    // Copy the advanced controls so all three runs are comparable.
    const topK = Number(els.topK.value);
    const seedRaw = els.seedValue.value.trim();

    const started = performance.now();
    const results = await Promise.all(temps.map(async (temperature, index) => {
      const card = cards[index];
      try {
        const result = await api("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            seed: seedText,
            temperature,
            num_generate: Math.round(Number(els.numGenerate.value)),
            top_k: topK > 0 ? topK : null,
            greedy: false,
            seed_value: seedRaw === "" ? null : Number(seedRaw),
            reflow: els.reflow.checked,
          }),
        });
        card.classList.remove("loading");
        const body = card.querySelector(".body");
        body.innerHTML = "";
        const span = document.createElement("span");
        span.className = "seed";
        span.textContent = result.text.slice(0, result.seed.length);
        body.append(span, document.createTextNode(result.text.slice(result.seed.length)));
        return result.text.length;
      } catch (error) {
        card.classList.remove("loading");
        card.querySelector(".body").innerHTML =
          `<span class="placeholder">${error.message}</span>`;
        return 0;
      }
    }));

    const elapsed = Math.round(performance.now() - started);
    els.compareStats.textContent =
      `${results.filter(Boolean).length}/${temps.length} runs · ${elapsed} ms total`;
    setBusy(false);
  });

  /* --------------------------------------------------------------- copy */

  els.copy.addEventListener("click", async () => {
    if (!state.lastText) return;
    try {
      await navigator.clipboard.writeText(state.lastText);
      const original = els.copy.textContent;
      els.copy.textContent = "Copied";
      setTimeout(() => { els.copy.textContent = original; }, 1400);
    } catch (_) {
      showError("The browser blocked clipboard access. Select the text and copy manually.");
    }
  });

  /* ------------------------------------------------------------ deep links */

  /** Apply ?seed=&temperature=&length=&top_k=&greedy=&seed_value= from the URL.
   *  Returns true when the link also asks to generate immediately (&run=1). */
  function applyUrlParams() {
    const params = new URLSearchParams(window.location.search);
    if (![...params.keys()].length) return false;

    if (params.has("seed")) els.seed.value = params.get("seed");

    if (params.has("temperature")) {
      const temp = Number(params.get("temperature"));
      if (Number.isFinite(temp)) {
        els.temperature.value = String(temp);
        els.temperatureReadout.textContent = temp.toFixed(2);
      }
    }
    if (params.has("length")) {
      const len = Number(params.get("length"));
      if (Number.isFinite(len)) {
        els.numGenerate.value = String(len);
        els.lengthReadout.textContent = String(Math.round(len));
      }
    }
    if (params.has("top_k")) {
      const k = Number(params.get("top_k"));
      if (Number.isFinite(k) && k >= 0) {
        els.topK.value = String(k);
        els.topKReadout.textContent = k > 0 ? `k = ${k}` : "off";
      }
    }
    if (params.has("seed_value")) els.seedValue.value = params.get("seed_value");
    if (params.get("greedy") === "1") {
      els.greedy.checked = true;
      els.greedy.dispatchEvent(new Event("change"));
    }
    if (params.get("advanced") === "1") els.advanced.open = true;
    if (params.get("reflow") === "0") els.reflow.checked = false;

    return params.get("run") === "1";
  }

  function renderPresets() {
    const holder = document.createElement("div");
    holder.className = "chips";
    holder.style.marginTop = "16px";
    holder.style.paddingTop = "14px";
    holder.style.borderTop = "1px solid rgba(217, 164, 65, 0.18)";

    PRESETS.forEach((preset) => {
      const link = document.createElement("a");
      link.className = "chip";
      link.textContent = preset.label;
      link.href = `?seed=${encodeURIComponent(preset.seed)}` +
                  `&temperature=${preset.temperature}` +
                  `&length=${preset.length}&run=1`;
      link.title = `${preset.seed} · T=${preset.temperature} · ${preset.length} chars`;
      holder.appendChild(link);
    });
    els.form.appendChild(holder);
  }

  /* ------------------------------------------------------------------ meta */

  const MODELS = [];

  function currentModel() {
    if (!els.model || !els.model.value) return null;
    return MODELS.find((m) => m.key === els.model.value) || null;
  }

  /** Apply a model's own limits and metadata to the controls. */
  function applyModel(model) {
    if (!model) return;

    // Transformers are far slower per character, so their ceiling is lower.
    const ceiling = Math.min(model.max_generate || 3000, 3000);
    els.numGenerate.max = String(ceiling);
    if (Number(els.numGenerate.value) > ceiling) {
      els.numGenerate.value = String(ceiling);
      els.lengthReadout.textContent = String(ceiling);
    }

    if (model.parameter_count) {
      $("badge-params").textContent =
        `${(model.parameter_count / 1e6).toFixed(2)}M parameters`;
    }
    if (model.vocab_size) {
      $("badge-vocab").textContent = `${model.vocab_size} characters`;
      $("note-vocab").textContent = String(model.vocab_size);
      els.topK.max = String(model.vocab_size);
    }
    if (model.context_length) {
      $("badge-context").textContent = `Context ${model.context_length}`;
    }
    const arch = model.architecture === "transformer" ? "Transformer" : "RNN";
    els.modelBlurb.textContent =
      `${arch} · ${model.model_file}` +
      (model.blurb ? ` — ${model.blurb}` : "") +
      (model.training_loss ? ` · val loss ${model.training_loss.toFixed(3)}` : "");
  }

  async function loadMeta() {
    try {
      const meta = await api("/api/meta");

      MODELS.length = 0;
      MODELS.push(...(meta.models || []));

      // The selector only earns its place when there is something to compare.
      if (MODELS.length > 1 && els.model) {
        els.model.innerHTML = "";
        MODELS.forEach((m) => {
          const option = document.createElement("option");
          option.value = m.key;
          option.textContent = m.label;
          els.model.appendChild(option);
        });
        els.model.value = meta.default_model || MODELS[0].key;
        els.modelField.hidden = false;
      }

      els.temperature.value = String(meta.default_temperature);
      els.temperatureReadout.textContent = meta.default_temperature.toFixed(2);
      els.numGenerate.value = String(meta.default_num_generate);
      els.lengthReadout.textContent = String(meta.default_num_generate);
      state.maxGenerate = meta.max_generate;

      applyModel(currentModel() || MODELS[0]);
    } catch (error) {
      showError(`Could not reach the generation service: ${error.message}`);
    }
  }

  if (els.model) {
    els.model.addEventListener("change", () => {
      applyModel(currentModel());
      // Re-run so the comparison is immediate rather than requiring a click.
      if (state.lastText) runGeneration();
    });
  }

  /* ------------------------------------------------- cross-model comparison */

  function modelCard(result) {
    const card = document.createElement("div");
    card.className = `compare-card ${result.architecture}`;

    const header = document.createElement("header");
    const name = document.createElement("span");
    name.textContent = result.label || result.model;
    const timing = document.createElement("span");
    timing.className = "temp";
    timing.textContent = result.elapsed_ms != null
      ? `${(result.elapsed_ms / 1000).toFixed(1)} s`
      : "—";
    header.append(name, timing);

    const meta = document.createElement("div");
    meta.className = "meta";
    const arch = document.createElement("span");
    arch.className = "arch";
    arch.textContent = result.architecture;
    meta.appendChild(arch);
    if (result.parameter_count) {
      const p = document.createElement("span");
      p.textContent = `${(result.parameter_count / 1e6).toFixed(2)}M params`;
      meta.appendChild(p);
    }
    if (result.context_length) {
      const c = document.createElement("span");
      c.textContent = `context ${result.context_length}`;
      meta.appendChild(c);
    }
    const t = document.createElement("span");
    t.textContent = `T = ${result.temperature.toFixed(2)}`;
    meta.appendChild(t);

    const body = document.createElement("div");
    body.className = "body";

    if (result.error) {
      const err = document.createElement("div");
      err.className = "error";
      err.textContent = result.error;
      body.appendChild(err);
    } else {
      const seedSpan = document.createElement("span");
      seedSpan.className = "seed";
      seedSpan.textContent = result.text.slice(0, currentSeedLength(result.text));
      body.append(seedSpan,
                  document.createTextNode(result.text.slice(seedSpan.textContent.length)));
    }

    card.append(header, meta, body);
    return card;
  }

  /** The generated text always begins with the cleaned seed. */
  function currentSeedLength(text) {
    const seed = els.seed.value.trim();
    if (text.startsWith(seed)) return seed.length;
    // Seeding strips characters the model does not know, so fall back to
    // matching as long a prefix as the two share.
    let n = 0;
    while (n < seed.length && n < text.length && seed[n] === text[n]) n += 1;
    return n;
  }

  async function runModelComparison() {
    if (state.busy) return;

    const seedText = els.seed.value.trim();
    if (!seedText) {
      showError("Enter a seed word or phrase first.");
      return;
    }
    if (MODELS.length < 2) {
      showError("Only one model is available. Add another .h5 to models/ and restart.");
      return;
    }

    showError("");
    setBusy(true);
    els.modelGrid.innerHTML = "";
    MODELS.forEach((m) => {
      const card = document.createElement("div");
      card.className = `compare-card ${m.architecture} loading`;
      const header = document.createElement("header");
      const name = document.createElement("span");
      name.textContent = m.label;
      const timing = document.createElement("span");
      timing.className = "temp";
      timing.textContent = "writing…";
      header.append(name, timing);
      const body = document.createElement("div");
      body.className = "body";
      const ph = document.createElement("span");
      ph.className = "placeholder";
      ph.textContent = `Generating with ${m.label}…`;
      body.appendChild(ph);
      card.append(header, body);
      els.modelGrid.appendChild(card);
    });

    try {
      const payload = {
        ...buildPayload(seedText, Number(els.temperature.value),
                        Number(els.numGenerate.value)),
        models: MODELS.map((m) => m.key),
      };
      const data = await api("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      els.modelGrid.innerHTML = "";
      data.results.forEach((r) => els.modelGrid.appendChild(modelCard(r)));
    } catch (error) {
      showError(error.message);
      els.modelGrid.innerHTML = "";
    } finally {
      setBusy(false);
    }
  }

  if (els.compareModels) {
    els.compareModels.addEventListener("click", runModelComparison);
  }

  /* ------------------------------------------------------------------ boot */

  async function boot() {
    renderPresets();
    await loadMeta();

    if (applyUrlParams()) await runGeneration();

    // Expose a small handle for automated checks and live console demos.
    window.vernebot = { runGeneration, runModelComparison, state, els, MODELS };
  }

  boot();
})();
