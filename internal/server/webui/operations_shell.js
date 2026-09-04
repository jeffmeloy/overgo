(function () {
  "use strict";

  const overgo = window.overgo;
  const { api, el, fmt } = overgo;
  const operations = new Map();
  let selected = "";
  let detailRequest = null;

  function terminal(state) {
    return state === "completed" || state === "cancelled" || state === "failed";
  }

  function routeOperation() {
    return new URL(window.location.href).searchParams.get("operation") || "";
  }

  function updateRoute(id, replace) {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set("operation", id);
    else url.searchParams.delete("operation");
    window.history[replace ? "replaceState" : "pushState"](null, "", url);
  }

  const artifactLink = window.overgo.artifactLink;

  function operationProgress(status) {
    const progress = status && status.progress || {};
    return progress.total == null ? String(progress.completed || 0) : progress.completed + " / " + progress.total;
  }

  function renderStrip(host) {
    const values = [...operations.values()];
    const active = values.filter((item) => !terminal(item.state)).length;
    const blocked = values.filter((item) => item.state === "blocked").length;
    const failed = values.filter((item) => item.state === "failed").length;
    const strip = el("div", { class: "operation-strip" },
      el("span", { class: "operation-strip-label", text: "Operations" }),
      el("span", { class: "note", text: active + " active / " + blocked + " blocked / " + failed + " failed" }));
    for (const item of values) {
      strip.appendChild(el("button", {
        class: "operation-chip " + item.state + (selected === item.id ? " active" : ""),
        text: item.state + " / " + fmt.shortID(item.id),
        title: item.task + " / " + item.id,
        onclick: () => selectOperation(host, item.id, false),
      }));
    }
    host.firstElementChild.replaceWith(strip);
  }

  async function cancelOperation(host, id) {
    try {
      await api.post("/operations/cancel", { id });
    } catch (err) {
      renderDetailError(host, err);
    }
  }

  async function decideOperation(host, id, tool, answer) {
    try {
      // A decision must name the advertised approval request, so read the
      // pending-decision view and grant exactly what it advertises now.
      const view = await api.get("/operations/decisions");
      const blocked = (view.blocked || []).find((item) => item.operation === id);
      const action = blocked && (blocked.actions || []).find((item) => item.code === tool);
      if (!action || !action.request) throw new Error("operation no longer advertises this decision");
      await api.post("/operations/decision", { operation: id, tool, answer, request: action.request });
      await loadDetail(host, id);
    } catch (err) {
      renderDetailError(host, err);
    }
  }

  function renderDetailError(host, err) {
    const detail = host.lastElementChild;
    detail.hidden = false;
    detail.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
  }

  function renderDetail(host, projection) {
    const detail = host.lastElementChild;
    const status = projection.operation || operations.get(projection.id) || {};
    if (status.id) operations.set(status.id, status);
    const summary = projection.summary || {};
    const usage = summary.usage || {};
    const resources = summary.resources || {};
    const head = el("div", { class: "operation-detail-head" },
      el("span", { class: "tag " + (status.state === "completed" ? "user_defined" : "control"), text: status.state || "durable" }),
      el("strong", { class: "mono", text: projection.id }),
      el("span", { class: "grow" }),
      el("button", { class: "btn alt", text: "Close", onclick: () => selectOperation(host, "", false) }));
    if (status.id && !terminal(status.state) && status.state !== "blocked") {
      head.insertBefore(el("button", { class: "btn alt", text: "Cancel", onclick: () => cancelOperation(host, status.id) }), head.lastElementChild);
    }

    const progress = el("progress", { class: "workflow-progress", value: (status.progress && status.progress.completed) || 0 });
    if (status.progress && status.progress.total != null) progress.max = status.progress.total;
    else progress.style.display = "none";
    const stats = el("div", { class: "operation-detail-grid" },
      overgo.stat("Task", status.task || "durable evidence"),
      overgo.stat("Progress", operationProgress(status)),
      overgo.stat("Stage attempts", summary.stage_attempts || 0),
      overgo.stat("Serving attempts", summary.serving_attempts || 0),
      overgo.stat("Input tokens", fmt.grouped(usage.input_tokens || 0)),
      overgo.stat("Output tokens", fmt.grouped(usage.output_tokens || 0)),
      overgo.stat("Peak host", fmt.bytes(resources.peak_host_bytes || 0)),
      overgo.stat("Peak device", fmt.bytes(resources.peak_device_bytes || 0)),
      overgo.stat("Host to device", fmt.bytes(resources.host_to_device_bytes || 0)),
      overgo.stat("Device to host", fmt.bytes(resources.device_to_host_bytes || 0)));

    const notices = el("div");
    if (status.failure) notices.appendChild(overgo.errorBanner(status.failure));
    if (status.recovery) {
      const actions = el("div", { class: "row" }, el("span", { text: status.recovery.reason }));
      for (const action of status.recovery.actions || []) actions.append(
        el("button", { class: "btn", text: "Grant " + action.code, onclick: () => decideOperation(host, status.id, action.code, "grant") }),
        el("button", { class: "btn alt", text: "Decline", onclick: () => decideOperation(host, status.id, action.code, "decline") }));
      notices.append(el("div", { class: "section-title", text: "Recovery decision" }), actions);
    }

    const links = el("div", { class: "row" });
    if (status.run) links.appendChild(artifactLink(status.run, "run " + fmt.shortID(status.run)));
    for (const output of status.outputs || []) links.appendChild(artifactLink(output, "output " + fmt.shortID(output)));
    for (const document of projection.interactions || []) {
      const trace = document.value && document.value.trace;
      if (trace) links.appendChild(artifactLink(trace, "trace " + fmt.shortID(trace)));
    }

    const attemptTable = el("table", { class: "grid" });
    attemptTable.appendChild(el("tr", {}, el("th", { text: "attempt" }), el("th", { text: "outcome" }),
      el("th", { text: "duration" }), el("th", { text: "evidence" })));
    for (const document of projection.serving || []) {
      const attempt = document.value;
      attemptTable.appendChild(el("tr", {}, el("td", { class: "mono", text: attempt.attempt }),
        el("td", { text: attempt.outcome }),
        el("td", { class: "mono", text: (Number(attempt.measured_ns || 0) / 1e6).toFixed(2) + " ms" }),
        el("td", {}, artifactLink(document.id))));
    }
    for (const attempt of status.attempts || []) attemptTable.appendChild(el("tr", {},
      el("td", { class: "mono", text: "operation" }), el("td", { text: "recorded" }), el("td", {}),
      el("td", {}, artifactLink(attempt))));

    const stageTable = el("table", { class: "grid" });
    stageTable.appendChild(el("tr", {}, el("th", { text: "stage" }), el("th", { text: "state" }),
      el("th", { text: "attempt" }), el("th", { text: "failure" })));
    for (const document of projection.stages || []) {
      const stage = document.value;
      stageTable.appendChild(el("tr", {}, el("td", { class: "mono", text: stage.node }),
        el("td", { text: stage.state }), el("td", { class: "mono", text: stage.attempt }),
        el("td", { text: stage.failure || "" })));
    }

    const decisionTable = el("table", { class: "grid" });
    decisionTable.appendChild(el("tr", {}, el("th", { text: "answer" }), el("th", { text: "tool" }),
      el("th", { text: "arguments" }), el("th", { text: "evidence" })));
    for (const document of projection.decisions || []) {
      const decision = document.value;
      decisionTable.appendChild(el("tr", {}, el("td", { text: decision.answer }), el("td", { text: decision.tool }),
        el("td", { class: "mono", text: (decision.arguments || []).join(" ") }),
        el("td", {}, artifactLink(document.id))));
    }

    detail.hidden = false;
    detail.replaceChildren(head, progress, stats, notices,
      el("div", { class: "section-title", text: "Outputs and traces" }), links,
      el("div", { class: "operation-evidence-columns" },
        el("div", {}, el("div", { class: "section-title", text: "Attempts" }), attemptTable),
        el("div", {}, el("div", { class: "section-title", text: "Stages" }), stageTable),
        el("div", {}, el("div", { class: "section-title", text: "Decisions" }), decisionTable)));
    renderStrip(host);
  }

  async function loadDetail(host, id) {
    if (detailRequest) detailRequest.abort();
    const controller = new AbortController();
    detailRequest = controller;
    try {
      const projection = await api.get("/operations/evidence?id=" + encodeURIComponent(id), { signal: controller.signal });
      if (selected === id) renderDetail(host, projection);
    } catch (err) {
      if (!err || err.name !== "AbortError") renderDetailError(host, err);
    } finally {
      if (detailRequest === controller) detailRequest = null;
    }
  }

  function selectOperation(host, id, replace) {
    selected = id;
    updateRoute(id, replace);
    renderStrip(host);
    if (!id) {
      if (detailRequest) detailRequest.abort();
      host.lastElementChild.hidden = true;
      host.lastElementChild.replaceChildren();
      return;
    }
    loadDetail(host, id);
  }

  function start() {
    const host = document.getElementById("global-operation-shell");
    if (!host) return;
    host.replaceChildren(el("div", { class: "operation-strip" }), el("div", { class: "operation-detail", hidden: true }));
    selected = routeOperation();
    renderStrip(host);
    if (selected) loadDetail(host, selected);
    overgo.runtimeEvents.subscribe((name, value) => {
      if (name === "operation.snapshot") {
        operations.clear();
        for (const item of value || []) operations.set(item.id, item);
        renderStrip(host);
        if (!selected) {
          host.lastElementChild.hidden = true;
          host.lastElementChild.replaceChildren();
        }
      }
      if (name === "operation") {
        operations.set(value.status.id, value.status);
        renderStrip(host);
        if (selected === value.status.id) {
          loadDetail(host, selected);
        }
      }
      if (name === "stream.error") renderDetailError(host, value);
    });
    window.addEventListener("popstate", () => {
      const id = routeOperation();
      if (id !== selected) selectOperation(host, id, true);
    });
    const key = document.getElementById("api-key");
    if (key) key.addEventListener("change", () => overgo.runtimeEvents.restart());
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
