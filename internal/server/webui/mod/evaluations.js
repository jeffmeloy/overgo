(function () {
  "use strict";

  const artifactLink = (overgo, id, label) => overgo.artifactLink(id, label, true);

  function metricTable(overgo, metrics) { return overgo.table(["metric", "value", "direction"], (metrics || []).map((metric) => [metric.name, String(metric.value) + (metric.unit ? " " + metric.unit : ""), metric.direction || "-"]), "metric-grid"); }
  function valueText(value) {
    if (value == null) return "-";
    return typeof value === "object" ? JSON.stringify(value) : String(value);
  }
  function recordTable(overgo, values) {
    return overgo.table(["field", "value"], Object.entries(values || {}).map(([name, value]) => [name, valueText(value)]));
  }

  window.overgo.registerTab({
    id: "evaluations",
    async mount(panel, overgo) {
      const { api, el, clear, fmt } = overgo;
      clear(panel);
      const model = el("select", { class: "text", "aria-label": "model" });
      const suiteHost = el("div", { class: "control-grid" });
      const run = el("button", { class: "btn", text: "Run" });
      const cancel = el("button", { class: "btn alt", text: "Cancel", style: "display:none" });
      const status = el("div", { class: "note" });
      const progress = el("progress", { class: "workflow-progress", value: 0, max: 1, style: "display:none" });
      const liveMetrics = el("div");
      const matrix = el("div");
      const history = el("div");
      const detail = el("div");
      panel.append(
        el("div", { class: "section-title", text: "Evaluation campaigns" }),
        el("div", { class: "row" }, model, run, cancel), suiteHost, status, progress, liveMetrics,
        el("div", { class: "section-title", text: "Available plans" }), matrix,
        el("div", { class: "section-title", text: "Previous runs" }), history, detail);

      let capabilities;
      try { capabilities = await api.get("/evaluations/capabilities"); }
      catch (err) {
        status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
        run.disabled = true;
        return;
      }
      const models = [...new Set(capabilities.map((item) => item.model))];
      for (const id of models) model.appendChild(el("option", { value: id, text: fmt.shortID(id) }));
      const fields = new Map();

      function selectedCapabilities() { return capabilities.filter((item) => item.model === model.value); }

      function renderCapabilities() {
        fields.clear();
        suiteHost.replaceChildren();
        const table = el("table", { class: "grid" }, overgo.headerRow(["select", "kind", "source", "cases", "plan"]));
        for (const capability of selectedCapabilities()) {
          const input = el("input", { type: "checkbox", value: capability.suite.plan });
          fields.set(capability.suite.plan, input);
          table.appendChild(overgo.tableRow([input, el("span", { text: capability.suite.kind }), el("span", { text: capability.suite.source }),
            String(capability.suite.cases), artifactLink(overgo, capability.suite.plan)]));
        }
        matrix.replaceChildren(table);
      }

      function renderOperation(current) {
        status.textContent = current.state + (current.run ? " / " + fmt.shortID(current.run) : "") +
          (current.failure ? " / " + current.failure : "");
        const total = current.progress && current.progress.total;
        progress.style.display = total ? "" : "none";
        if (total) {
          progress.max = total;
          progress.value = current.progress.completed;
        }
        liveMetrics.replaceChildren(...((current.metrics || []).length ? [metricTable(overgo, current.metrics)] : []));
      }

      async function showReport(entry) {
        if (!entry.report) return;
        detail.replaceChildren(el("div", { class: "note", text: "loading " + fmt.shortID(entry.report) }));
        try {
          const [report, failures, runDetail] = await Promise.all([
            api.get("/evaluations/report?id=" + encodeURIComponent(entry.report)),
            api.get("/evaluations/failures?id=" + encodeURIComponent(entry.report)),
            api.get("/runs?id=" + encodeURIComponent(entry.run)),
          ]);
          const observations = report.body && Array.isArray(report.body.observations) ? report.body.observations : [];
          detail.replaceChildren(
            el("div", { class: "section-title", text: "Report" }),
            el("div", { class: "row artifact-links" }, artifactLink(overgo, entry.run, "run " + fmt.shortID(entry.run)),
              artifactLink(overgo, entry.report, "report " + fmt.shortID(entry.report))),
            el("div", { class: "statgrid" }, overgo.stat("Failures", failures.length),
              overgo.stat("Inputs", (runDetail.inputs || []).length), overgo.stat("Outputs", (runDetail.outputs || []).length)),
            metricTable(overgo, entry.metrics),
            ...failures.map((failure) => recordTable(overgo, failure.observation)),
            ...observations.map((observation) => recordTable(overgo, observation)));
        } catch (err) { detail.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }

      let baseline = null;
      async function compare(entry) {
        if (!baseline) {
          baseline = entry;
          status.textContent = "baseline / " + fmt.shortID(entry.evaluation);
          return;
        }
        try {
          const comparison = await api.get("/evaluations/compare?left=" + encodeURIComponent(baseline.evaluation) +
            "&right=" + encodeURIComponent(entry.evaluation));
          const table = overgo.table(["metric", "baseline", "current", "delta", "result"], (comparison.metrics || []).map((metric) => [
            metric.name, String(metric.left), String(metric.right), String(metric.delta), el("span", { text: metric.improved ? "improved" : "not improved" })]));
          detail.replaceChildren(el("div", { class: "section-title", text: "Comparison" }), table);
          baseline = null;
        } catch (err) { detail.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }

      async function loadHistory() {
        if (!model.value) return;
        try {
          const entries = await api.get("/evaluations/history?model=" + encodeURIComponent(model.value));
          const table = overgo.table(["outcome", "commit", "metrics", "run", "actions"], entries.map((entry) => [entry.outcome, (entry.code_commit || "").slice(0, 10),
            el("span", { text: (entry.metrics || []).map((item) => item.name + "=" + item.value + " " + item.direction).join(", ") || "-" }),
            artifactLink(overgo, entry.run),
            el("span", { class: "row" }, entry.report ? el("button", { class: "btn alt", text: "Inspect", onclick: () => showReport(entry) }) : null,
              entry.evaluation ? el("button", { class: "btn alt", text: "Compare", onclick: () => compare(entry) }) : null)]));
          history.replaceChildren(table);
        } catch (err) { history.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }

      let operation = null;
      cancel.addEventListener("click", async () => {
        if (operation) await api.post("/operations/cancel", { id: operation });
      });
      run.addEventListener("click", async () => {
        const plans = [...fields].filter(([, input]) => input.checked).map(([plan]) => plan);
        if (!plans.length) {
          status.textContent = "select an evaluation";
          return;
        }
        run.disabled = true;
        cancel.style.display = "";
        try {
          const accepted = await api.post("/evaluations/run", { model: model.value, plans });
          operation = accepted.operation;
          const completed = await overgo.waitOperation(operation, renderOperation);
          renderOperation(completed);
          await loadHistory();
        } catch (err) { status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); } finally {
          operation = null;
          run.disabled = false;
          cancel.style.display = "none";
        }
      });
      model.addEventListener("change", () => { renderCapabilities(); loadHistory(); });
      renderCapabilities();
      await loadHistory();
    },
  });
})();
