(function () {
  "use strict";

  const artifactLink = (overgo, id, label) => overgo.artifactLink(id, label, true);

  async function readTrace(overgo, id) {
    const trace = await overgo.api.get("/artifacts/content?id=" + encodeURIComponent(id));
    return trace && (Array.isArray(trace.dpo) || Array.isArray(trace.grpo)) ? trace : null; }

  async function traceFromRun(overgo, run) {
    for (const edge of run.children || []) {
      if (!String(edge.child).startsWith("evidence:")) continue;
      try {
        const trace = await readTrace(overgo, edge.child);
        if (trace && trace.run === run.id) return { id: edge.child, trace };
      } catch (_) { /* unrelated evidence */ }
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
        artifactLink(overgo, run || trace.run, "run " + overgo.fmt.shortID(run || trace.run)),
        checkpoint ? artifactLink(overgo, checkpoint, "checkpoint " + overgo.fmt.shortID(checkpoint)) : null,
        artifactLink(overgo, id, "trace " + overgo.fmt.shortID(id))),
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
        artifactLink(overgo, baseline.checkpoint, "baseline " + overgo.fmt.shortID(baseline.checkpoint)),
        artifactLink(overgo, current.checkpoint, "current " + overgo.fmt.shortID(current.checkpoint))), table);
  }

  window.overgo.trainingEvidence = {
    async renderOperation(host, operation, overgo) {
      const traceID = (operation.outputs || []).find((id) => String(id).startsWith("evidence:"));
      const checkpoint = (operation.outputs || []).find((id) => String(id).startsWith("checkpoint:"));
      if (!traceID) return;
      const trace = await readTrace(overgo, traceID);
      if (trace) renderTrace(overgo, host, { id: traceID, trace, checkpoint, run: operation.run });
    },
  };

  window.overgo.registerTab({
    id: "runs",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      const limit = 50;
      let cursor = "";
      let nextCursor = "";
      let start = 0;
      let prior = [];
      let baseline = null;

      clear(panel);
      const status = el("span", { class: "note" });
      const prev = el("button", { class: "btn alt", onclick: () => {
        const page = prior.pop() || { cursor: "", start: 0 };
        cursor = page.cursor; start = page.start; load();
      }}, "prev");
      const next = el("button", { class: "btn alt", onclick: () => {
        prior.push({ cursor, start }); cursor = nextCursor; start += limit; load();
      }}, "next");
      panel.append(
        el("div", { class: "section-title", text: "Training runs" }),
        el("div", { class: "row mb-10" }, prev, next, status));
      const host = el("div");
      const detail = el("div");
      panel.append(host, detail);

      async function loadDetail(id) {
        try {
          const run = await overgo.api.get("/runs?id=" + encodeURIComponent(id));
          const found = await traceFromRun(overgo, run);
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
            el("div", { class: "row" }, artifactLink(overgo, run.id), pin), evidence);
          if (record) renderTrace(overgo, evidence, record, baseline && baseline.run !== record.run ? baseline : null);
        } catch (err) { detail.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }

      function outcomeClass(outcome) { return outcome === "succeeded" ? "user_defined" : (outcome === "failed" ? "control" : ""); }
      async function load() {
        host.replaceChildren(el("div", { class: "note", text: "loading /runs" }));
        let data;
        try {
          const query = new URLSearchParams({ limit: String(limit) });
          if (cursor) query.set("cursor", cursor);
          data = await overgo.api.get("/runs?" + query);
        } catch (err) { const message = err.status === 501 ? "Run browsing is not configured." : overgo.friendlyError(err); host.replaceChildren(overgo.errorBanner(message)); return; }
        nextCursor = data.next || "";
        const first = data.count === 0 ? 0 : start + 1;
        const last = start + data.runs.length;
        status.textContent = first + "-" + last + " of " + fmt.grouped(data.count) + " runs";
        prev.disabled = prior.length === 0;
        next.disabled = !nextCursor;

        const table = el("table", { class: "grid" }, overgo.headerRow(["outcome", "recipe", "commit", "wall", "heaviest phases", "in/out"]));
        for (const run of data.runs) {
          const phases = [...(run.phases || [])].sort((a, b) => b.ms - a.ms).slice(0, 3)
            .map((phase) => phase.phase + " " + phase.ms.toFixed(0) + "ms").join(", ");
          table.appendChild(overgo.tableRow([el("span", { class: "tag " + outcomeClass(run.outcome), text: run.outcome }),
            el("button", { class: "link-button mono", title: run.recipe, text: fmt.shortID(run.recipe), onclick: () => loadDetail(run.id) }),
            (run.code_commit || "").slice(0, 10) || "-", run.measured_ms ? run.measured_ms.toFixed(0) + " ms" : "-",
            el("span", { class: "dim", text: phases || "-" }), run.inputs + "/" + run.outputs]));
        }
        host.replaceChildren(table);
      }
      await load();
    },
  });
})();
