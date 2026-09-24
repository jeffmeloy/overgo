(function () {
  "use strict";


  function metricTable(overgo, metrics) { return overgo.table(["metric", "value", "direction"], (metrics || []).map((metric) => [metric.name, String(metric.value) + (metric.unit ? " " + metric.unit : ""), metric.direction || "-"]), "metric-grid"); }
  function valueText(value) { if (value == null) return "-"; return typeof value === "object" ? JSON.stringify(value) : String(value); }
  function recordTable(overgo, values) {
    return overgo.table(["field", "value"], Object.entries(values || {}).map(([name, value]) => [name, valueText(value)]));
  }

  window.overgo.registerTab({
    id: "evaluations",
    async mount(panel, overgo) {
      const { api, el, fmt } = overgo;
      const model = el("select", { class: "text", "aria-label": "model" });
      const suiteHost = el("div", { class: "control-grid" });
      const run = el("button", { class: "btn", text: "Run", disabled: true });
      const cancel = el("button", { class: "btn alt", text: "Cancel", hidden: true });
      const status = el("div", { class: "note" });
      const progress = el("progress", { class: "workflow-progress", value: 0, max: 1, hidden: true });
      const liveMetrics = el("div");
      const matrix = el("div");
      const history = el("div");
      const detail = el("div");
      panel.append(
        el("div", { class: "section-title", text: "Evaluation campaigns" }),
        el("div", { class: "row" }, model, run, cancel), suiteHost, status, progress, liveMetrics,
        el("div", { class: "section-title", text: "Available plans" }), matrix,
        el("div", { class: "section-title", text: "Previous runs" }), history, detail);

      let capabilities = [];
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
            String(capability.suite.cases), overgo.artifactLink(capability.suite.plan, null, true)]));
        }
        matrix.replaceChildren(table);
      }

      function renderOperation(current) {
        status.textContent = current.state + (current.run ? " / " + fmt.shortID(current.run) : "") +
          (current.failure ? " / " + current.failure : "");
        const total = current.progress && current.progress.total;
        progress.hidden = !total;
        if (total) { progress.max = total; progress.value = current.progress.completed; }
        liveMetrics.replaceChildren(...((current.metrics || []).length ? [metricTable(overgo, current.metrics)] : []));
      }

      // The detail shows the newest inspection or comparison; an earlier one still reading is aborted.
      let inspected = null, compared = null;
      const showDetail = overgo.read(detail, (signal) => inspected ? Promise.all([
        api.get("/evaluations/report?id=" + encodeURIComponent(inspected.report), { signal }),
        api.get("/evaluations/failures?id=" + encodeURIComponent(inspected.report), { signal }),
        api.get("/runs?id=" + encodeURIComponent(inspected.run), { signal }),
      ]) : api.get("/evaluations/compare?left=" + encodeURIComponent(compared[0].evaluation) + "&right=" + encodeURIComponent(compared[1].evaluation), { signal }),
      (answer) => inspected ? renderReport(inspected, ...answer) : renderComparison(answer));
      function showReport(entry) {
        if (!entry.report) return;
        inspected = entry;
        showDetail();
      }
      function renderReport(entry, report, failures, runDetail) {
        const observations = report.body && Array.isArray(report.body.observations) ? report.body.observations : [];
        detail.replaceChildren(
          el("div", { class: "section-title", text: "Report" }),
          el("div", { class: "row artifact-links" }, overgo.artifactLink(entry.run, "run " + fmt.shortID(entry.run), true),
            overgo.artifactLink(entry.report, "report " + fmt.shortID(entry.report), true)),
          el("div", { class: "statgrid" }, overgo.stat("Failures", failures.length),
            overgo.stat("Inputs", (runDetail.inputs || []).length), overgo.stat("Outputs", (runDetail.outputs || []).length)),
          metricTable(overgo, entry.metrics),
          ...failures.map((failure) => recordTable(overgo, failure.observation)),
          ...observations.map((observation) => recordTable(overgo, observation)));
      }

      let baseline = null;
      function compare(entry) {
        if (!baseline) { baseline = entry; status.textContent = "baseline / " + fmt.shortID(entry.evaluation); return; }
        inspected = null;
        compared = [baseline, entry];
        baseline = null;
        showDetail();
      }
      function renderComparison(comparison) {
        const table = overgo.table(["metric", "baseline", "current", "delta", "result"], (comparison.metrics || []).map((metric) => [
          metric.name, String(metric.left), String(metric.right), String(metric.delta), el("span", { text: metric.improved ? "improved" : "not improved" })]));
        detail.replaceChildren(el("div", { class: "section-title", text: "Comparison" }), table);
      }

      const loadHistory = overgo.read(history, (signal) => {
        if (!model.value) return [];
        return api.get("/evaluations/history?model=" + encodeURIComponent(model.value), { signal });
      }, (entries) => {
        const table = overgo.table(["outcome", "commit", "metrics", "run", "actions"], entries.map((entry) => [entry.outcome, (entry.code_commit || "").slice(0, 10),
          el("span", { text: (entry.metrics || []).map((item) => item.name + "=" + item.value + " " + item.direction).join(", ") || "-" }),
          overgo.artifactLink(entry.run, null, true),
          el("span", { class: "row" }, entry.report ? el("button", { class: "btn alt", text: "Inspect", onclick: () => showReport(entry) }) : null,
            entry.evaluation ? el("button", { class: "btn alt", text: "Compare", onclick: () => compare(entry) }) : null)]));
        history.replaceChildren(table);
      }, { empty: (entries) => !entries.length && "No evaluation has run for this model yet." });

      let operation = null;
      cancel.addEventListener("click", async () => {
        if (operation) await overgo.cancelOperation(operation);
      });
      run.addEventListener("click", async () => {
        const plans = [...fields].filter(([, input]) => input.checked).map(([plan]) => plan);
        if (!plans.length) { status.textContent = "select an evaluation"; return; }
        run.disabled = true;
        cancel.hidden = false;
        try {
          const accepted = await api.post("/evaluations/run", { model: model.value, plans });
          operation = accepted.operation;
          const completed = await overgo.waitOperation(operation, renderOperation);
          renderOperation(completed);
          await loadHistory();
        } catch (err) { status.replaceChildren(overgo.failure(err)); } finally { operation = null; run.disabled = false; cancel.hidden = true; }
      });
      model.addEventListener("change", () => { renderCapabilities(); loadHistory(); });
      const loadCapabilities = overgo.read(status, (signal) => api.get("/evaluations/capabilities", { signal }), (value) => {
        capabilities = value;
        status.replaceChildren();
        for (const id of new Set(capabilities.map((item) => item.model))) model.appendChild(el("option", { value: id, text: fmt.shortID(id) }));
        run.disabled = false;
        renderCapabilities();
      }, { loading: "Loading evaluation plans…" });
      await loadCapabilities();
      if (capabilities.length) await loadHistory();
    },
  });
})();
