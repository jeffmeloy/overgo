(function () {
  "use strict";


  async function readTrace(overgo, id, signal) {
    const trace = await overgo.api.get(overgo.contentURL(id), { signal });
    return trace && (Array.isArray(trace.dpo) || Array.isArray(trace.grpo)) ? trace : null; }

  async function traceFromRun(overgo, run, signal) {
    for (const edge of run.children || []) {
      if (!String(edge.child).startsWith("evidence:")) continue;
      try {
        const trace = await readTrace(overgo, edge.child, signal);
        if (trace && trace.run === run.id) return { id: edge.child, trace };
      } catch (err) { if (err && err.name === "AbortError") throw err; /* otherwise unrelated evidence */ }
    }
    return null;
  }

  function seriesBlock(overgo, title, series) {
    const legend = overgo.el("div", { class: "series-legend" }, ...series.map((item) => {
      const swatch = overgo.el("i", { class: "swatch" }); swatch.style.background = item.color; // the series' own colour
      return overgo.el("span", {}, swatch, item.label); }));
    return overgo.el("section", { class: "evidence-block" },
      overgo.el("div", { class: "section-title", text: title }), legend,
      overgo.viz.signedSeries(series));
  }

  function pairInspector(overgo, observations) {
    const select = overgo.el("select", { class: "text", "aria-label": "observation step" });
    observations.forEach((observation, index) => select.appendChild(overgo.el("option", {
      value: index, text: "step " + observation.step,
    })));
    const detail = overgo.el("div");
    function render() {
      const observation = observations[Number(select.value) || 0];
      const table = overgo.table(["scorer", "chosen", "rejected", "margin"], [
        ["policy", observation.policy_chosen, observation.policy_rejected, observation.policy_margin],
        ["reference", observation.reference_chosen, observation.reference_rejected, observation.reference_margin],
      ]);
      detail.replaceChildren(
        overgo.el("div", { class: "statgrid" },
          overgo.stat("Relative margin", observation.relative_margin),
          overgo.stat("Chosen tokens", observation.chosen_tokens),
          overgo.stat("Rejected tokens", observation.rejected_tokens)), table);
    }
    select.addEventListener("change", render);
    render();
    return overgo.el("section", { class: "evidence-block" },
      overgo.el("div", { class: "section-title", text: "Chosen / rejected pair" }), select, detail);
  }

  function renderTrace(overgo, host, record, comparison) {
    const { trace, id, checkpoint, run } = record;
    const observations = trace.objective === "grpo" ? trace.grpo : trace.dpo;
    const colors = ["var(--acc)", "var(--amber)", "var(--ok)"];
    const objectiveBlocks = trace.objective === "grpo" ? [
      seriesBlock(overgo, "GRPO loss", [{ label: "loss", values: observations.map((row) => row.loss), color: colors[0] }]),
      seriesBlock(overgo, "Evaluator reward", [
        { label: "mean", values: observations.map((row) => row.mean_reward), color: colors[0] },
        { label: "dispersion", values: observations.map((row) => row.reward_dispersion), color: colors[1] },
      ]),
    ] : [
      seriesBlock(overgo, "DPO loss", [{ label: "loss", values: observations.map((row) => row.loss), color: colors[0] }]),
      seriesBlock(overgo, "Margin decomposition", [
        { label: "policy", values: observations.map((row) => row.policy_margin), color: colors[0] },
        { label: "reference", values: observations.map((row) => row.reference_margin), color: colors[1] },
        { label: "relative", values: observations.map((row) => row.relative_margin), color: colors[2] },
      ]),
    ];
    host.replaceChildren(
      overgo.el("div", { class: "row artifact-links" },
        overgo.artifactLink(run || trace.run, "run " + overgo.fmt.shortID(run || trace.run), true),
        checkpoint ? overgo.artifactLink(checkpoint, "checkpoint " + overgo.fmt.shortID(checkpoint), true) : null,
        overgo.artifactLink(id, "trace " + overgo.fmt.shortID(id), true)),
      overgo.el("div", { class: "evidence-grid" },
        ...objectiveBlocks,
        seriesBlock(overgo, "Optimizer health / gradient L2", [
          { label: "gradient L2", values: observations.map((row) => row.gradient_l2), color: colors[0] }]),
        seriesBlock(overgo, "Optimizer health / update L2", [
          { label: "update L2", values: observations.map((row) => row.update_l2), color: colors[1] }]),
        seriesBlock(overgo, "Optimizer health / learning rate", [
          { label: "learning rate", values: observations.map((row) => row.learning_rate), color: colors[2] }])),
      trace.objective === "dpo" ? pairInspector(overgo, observations) : null,
      comparison ? checkpointComparison(overgo, comparison, record) : null);
  }

  function checkpointComparison(overgo, baseline, current) {
    if (baseline.trace.objective !== current.trace.objective) return null;
    const rows = (trace) => trace.objective === "grpo" ? trace.grpo : trace.dpo;
    const left = rows(baseline.trace).at(-1);
    const right = rows(current.trace).at(-1);
    const fields = current.trace.objective === "grpo" ?
      ["loss", "mean_reward", "reward_dispersion", "gradient_l2", "update_l2"] :
      ["loss", "policy_margin", "reference_margin", "relative_margin", "gradient_l2", "update_l2"];
    const table = overgo.table(["measurement", "baseline", "current"], fields.map((field) => [field, String(left[field]), String(right[field])]));
    return overgo.el("section", { class: "evidence-block" },
      overgo.el("div", { class: "section-title", text: "Checkpoint comparison" }),
      overgo.el("div", { class: "row artifact-links" },
        overgo.artifactLink(baseline.checkpoint, "baseline " + overgo.fmt.shortID(baseline.checkpoint), true),
        overgo.artifactLink(current.checkpoint, "current " + overgo.fmt.shortID(current.checkpoint), true)), table);
  }

  // lengthBins: the token-length distribution's bar count, fewer when lengths span less.
  const lengthBins = 8;
  function lengthDistribution(overgo, lengths) {
    if (!lengths?.length) return null; // a preview that reads only its page lists no lengths
    const low = Math.min(...lengths), high = Math.max(...lengths);
    const width = Math.max(1, Math.ceil((high - low + 1) / lengthBins));
    const counts = new Map();
    for (const length of lengths) { const bin = Math.floor((length - low) / width); counts.set(bin, (counts.get(bin) || 0) + 1); }
    const entries = [...counts.keys()].sort((a, b) => a - b).map((bin) => {
      const from = low + bin * width, to = Math.min(high, from + width - 1);
      return { label: (from === to ? String(from) : from + "–" + to) + " tokens", value: counts.get(bin), title: counts.get(bin) + " sequences" };
    });
    return overgo.el("section", { class: "evidence-block", "aria-label": "Token lengths" },
      overgo.el("div", { class: "section-title", text: "Token lengths · " + lengths.length + " sequences" }), overgo.viz.probBars(entries));
  }

  // trainingPreview: a dataset's records as the run's objective encodes them, before the run: each
  // sequence's tokens with the masked prompt prefix faint and the trained response marked, a record
  // pager, and every sequence's token length. request() is the run's task, recipe and input.
  function trainingPreview(host, request, overgo) {
    const { el, fmt } = overgo;
    const status = el("span", { class: "note", role: "status" });
    const body = el("div");
    const prev = el("button", { class: "btn alt", text: "Previous", disabled: true });
    const next = el("button", { class: "btn alt", text: "Next", disabled: true });
    let position = 0;
    const show = overgo.read(body, (signal) => overgo.api.post("/training/preview", { ...request(), position, limit: 1 }, { signal }), (preview) => {
      status.textContent = preview.records ? "record " + (position + 1) + " of " + fmt.grouped(preview.records) : "no records";
      prev.disabled = position === 0;
      next.disabled = position + 1 >= preview.records;
      body.replaceChildren(...preview.page.flatMap((record) => [
        el("div", { class: "section-title", text: record.id }),
        ...record.sequences.map((sequence) => el("section", { class: "evidence-block", "data-sequence": sequence.label },
          el("div", { class: "note", text: sequence.label + " · " + sequence.tokens.filter((token) => token.trained).length + " trained of " + sequence.tokens.length + " tokens" }),
          el("div", { class: "token-run" }, ...sequence.tokens.map((token) => el("span", {
            class: token.trained ? "trained" : "masked", title: "id " + token.id + (token.trained ? " · trained" : " · masked prompt"), text: overgo.displayToken(token.piece),
          }))))),
      ]), lengthDistribution(overgo, preview.lengths) || "");
    }, { loading: "Encoding records…" });
    prev.addEventListener("click", () => { position--; show(); });
    next.addEventListener("click", () => { position++; show(); });
    const open = el("button", { class: "btn alt", text: "Preview as trained", onclick: () => { position = 0; show(); } });
    host.replaceChildren(el("div", { class: "row" }, open, prev, next, status), body);
  }

  // renderVerdict: a held-out verdict as the scorecard reads a trained model's quality: the base model's
  // score, the trained model's, and whether it beats the base, beside the trained model and the run.
  function renderVerdict(overgo, host, verdict, id, trained, run) {
    const { el, fmt } = overgo;
    const sign = { maximize: 1, minimize: -1 }[verdict.direction] || 0;
    // Below, level with, or beating the base, by the sign of the improvement in the metric's own direction.
    const outcome = ["below base", "level with base", "beats base"][Math.sign(sign * (verdict.candidate_value - verdict.baseline_value)) + 1];
    const metric = String(verdict.metric).replace(/[_-]/g, " ");
    // The label names the role; the short identity after it drops the kind the label already says.
    const link = (target, label) => overgo.artifactLink(target, label + " " + fmt.shortID(target).replace(/^[a-z]+:/, ""), true);
    host.replaceChildren(el("section", { class: "evidence-block", "aria-label": "Held-out verdict" },
      el("div", { class: "section-title", text: "Held-out " + metric }),
      el("div", { class: "statgrid" },
        overgo.stat("Base model", verdict.baseline_value.toFixed(3)),
        overgo.stat("Trained model", verdict.candidate_value.toFixed(3)),
        overgo.stat("Verdict", outcome)),
      el("div", { class: "row artifact-links" },
        link(run, "run"), trained ? link(trained, "trained model") : null, link(id, "verdict"))));
  }

  // renderSamples: what a run generated after the same held-out prompts, before it trained and as it went,
  // each prompt with the reply the conversation holds and the generations in step order beneath it.
  function renderSamples(overgo, host, samples) {
    const { el } = overgo;
    host.append(el("section", { class: "evidence-block", "aria-label": "Training samples" },
      el("div", { class: "section-title", text: "Samples while training" }),
      ...samples.strips.map((strip) => el("div", { "data-record": strip.record },
        el("div", { class: "note", text: "prompt" }),
        el("div", { class: "preview-text", text: strip.prompt }),
        el("div", { class: "note", text: "reply in the conversation" }),
        el("div", { class: "preview-text", text: strip.expected }),
        el("ol", {}, ...strip.samples.map((sample) => el("li", { "data-step": String(sample.step) },
          el("span", { class: "note", text: sample.step === 0 ? "before training " : "after " + sample.step + " updates " }),
          el("span", { class: "preview-text", text: sample.text }))))))));
  }

  window.overgo.trainingPreview = trainingPreview;
  window.overgo.trainingEvidence = {
    // renderLive: a running sampled run reports its strips so far as its text, and they stand on the page as each
    // round ends. The same text arrives on more than one event and is drawn once: the section that was drawn
    // remembers it, so a later run, which starts on an emptied host, is drawn afresh even when its first round
    // reads the same. A text that does not parse is left out: a throw here would end the page's event stream.
    renderLive(host, operation, overgo) {
      const text = operation.text, drawn = host.querySelector('[aria-label="Training samples"]');
      if (!text || (drawn && drawn.liveText === text)) return;
      let strips = null;
      try { strips = JSON.parse(text); } catch (_) { /* not strips: nothing to draw */ }
      if (!Array.isArray(strips)) return;
      host.replaceChildren();
      renderSamples(overgo, host, { strips });
      host.lastElementChild.liveText = text;
    },
    async renderOperation(host, operation, overgo) {
      const outputs = operation.outputs || [];
      const checkpoint = outputs.find((id) => String(id).startsWith("checkpoint:"));
      const verdictID = outputs.find((id) => String(id).startsWith("evaluation:"));
      if (verdictID) {
        // Both are read before the host is written, so the strip of the run that just ended stands until the
        // verdict and the published samples replace it together.
        const verdict = await overgo.api.get(overgo.contentURL(verdictID));
        // A sampled run also published what it generated as it trained, beside the verdict.
        const samplesID = outputs.find((id) => String(id).startsWith("evidence:"));
        const samples = samplesID ? await overgo.api.get(overgo.contentURL(samplesID)) : null;
        if (verdict && typeof verdict.baseline_value === "number" && typeof verdict.candidate_value === "number") renderVerdict(overgo, host, verdict, verdictID, checkpoint, operation.run);
        else host.replaceChildren();
        if (samples && Array.isArray(samples.strips)) renderSamples(overgo, host, samples);
        return;
      }
      const traceID = outputs.find((id) => String(id).startsWith("evidence:"));
      if (!traceID) return;
      const trace = await readTrace(overgo, traceID);
      if (trace) renderTrace(overgo, host, { id: traceID, trace, checkpoint, run: operation.run });
    },
  };

  window.overgo.registerTab({
    id: "runs",
    async mount(panel, overgo) {
      const { el, fmt } = overgo;
      const limit = 50;
      let cursor = "";
      let nextCursor = "";
      let start = 0;
      let prior = [];
      let baseline = null;

      const status = el("span", { class: "note" });
      const prev = el("button", { class: "btn alt", onclick: () => {
        const page = prior.pop() || { cursor: "", start: 0 };
        cursor = page.cursor; start = page.start; load();
      }}, "Previous");
      const next = el("button", { class: "btn alt", onclick: () => {
        prior.push({ cursor, start }); cursor = nextCursor; start += limit; load();
      }}, "Next");
      panel.append(
        el("div", { class: "section-title", text: "Training runs" }),
        el("div", { class: "row mb-s3" }, prev, next, status));
      const host = el("div");
      const detail = el("div");
      panel.append(host, detail);

      // The detail shows the newest chosen run; an earlier one still reading is aborted.
      let chosen = "";
      const showDetail = overgo.read(detail, async (signal) => {
        const run = await overgo.api.get("/runs?id=" + encodeURIComponent(chosen), { signal });
        return [run, await traceFromRun(overgo, run, signal)];
      }, ([run, found]) => {
        const checkpoint = (run.outputs || []).find((value) => String(value).startsWith("checkpoint:"));
        const record = found && { id: found.id, trace: found.trace, checkpoint, run: run.id };
        const evidence = el("div");
        const pin = record && el("button", { class: "btn alt", text: "Set comparison baseline", onclick: () => { baseline = record; pin.textContent = "Comparison baseline"; pin.disabled = true; } });
        detail.replaceChildren(
          el("div", { class: "section-title", text: "Run detail" }),
          el("div", { class: "statgrid" },
            overgo.stat("Outcome", run.outcome),
            overgo.stat("Inputs", (run.inputs || []).length),
            overgo.stat("Outputs", (run.outputs || []).length)),
          el("div", { class: "row" }, overgo.artifactLink(run.id, null, true), pin), evidence);
        if (record) renderTrace(overgo, evidence, record, baseline && baseline.run !== record.run ? baseline : null);
      }, { loading: "Loading the run…" });

      function outcomeClass(outcome) { return outcome === "succeeded" ? "ok" : (outcome === "failed" ? "caution" : ""); }
      const load = overgo.read(host, (signal) => {
        const query = new URLSearchParams({ limit: String(limit) });
        if (cursor) query.set("cursor", cursor);
        return overgo.api.get("/runs?" + query, { signal }).catch((err) => {
          if (err.status === 501) throw new Error("Run browsing is not configured on this server.");
          throw err;
        });
      }, (data) => {
        nextCursor = data.next || "";
        const first = data.count === 0 ? 0 : start + 1;
        const last = start + data.runs.length;
        status.textContent = first + "-" + last + " of " + fmt.grouped(data.count) + " runs";
        prev.disabled = prior.length === 0;
        next.disabled = !nextCursor;
        // One page needs no paging: the count says it all; no runs at all says so instead of empty headers.
        prev.hidden = next.hidden = prev.disabled && next.disabled;
        if (!data.count) {
          status.textContent = "No training runs yet. A run appears here once a training job finishes.";
          host.replaceChildren();
          return;
        }

        // A run reads as how it ended, how long it took and what it used and made; its commit is reference.
        const took = (ms) => ms < 1000 ? ms.toFixed(0) + " ms" : fmt.duration(ms);
        const table = el("table", { class: "grid" }, overgo.headerRow(["outcome", "recipe", "took", "slowest phases", "inputs and outputs", "commit"]));
        for (const run of data.runs) {
          const phases = [...(run.phases || [])].sort((a, b) => b.ms - a.ms).slice(0, 3)
            .map((phase) => phase.phase + " " + took(phase.ms)).join(", ");
          table.appendChild(overgo.tableRow([el("span", { class: "tag " + outcomeClass(run.outcome), text: run.outcome }),
            el("button", { class: "link-button mono", title: run.recipe, text: fmt.shortID(run.recipe), onclick: () => { chosen = run.id; showDetail(); } }),
            run.measured_ms ? took(run.measured_ms) : "-", el("span", { class: "dim", text: phases || "-" }),
            run.inputs + " in · " + run.outputs + " out", el("span", { class: "note mono", text: (run.code_commit || "").slice(0, 10) || "-" })]));
        }
        host.replaceChildren(table);
      }, { loading: "Loading runs…" });
      await load();
    },
  });
})();
