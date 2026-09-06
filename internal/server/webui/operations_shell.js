/* operations_shell.js: the operations strip under the header on every page: each runtime operation as a chip
   with progress, cancel, the live event tail and the durable receipt (/operations/evidence); the approvals
   count in the header that opens the inbox; and local chips a page adds (a model switch). */
(function () {
  "use strict";

  const overgo = window.overgo;
  const { api, el, fmt } = overgo;
  const operations = new Map();
  const tails = new Map(); // operation id -> the last events the stream reported
  let selected = "";
  let detailRequest = null;
  let shellHost = null;

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
        text: item.state + " / " + (item.local ? item.task : fmt.shortID(item.id)) + (item.progress && item.progress.total != null ? " · " + operationProgress(item) : ""),
        title: item.task + " / " + item.id,
        onclick: () => selectOperation(host, item.id, false),
      }));
    }
    host.firstElementChild.replaceWith(strip);
    // The approvals count: every blocked operation waiting on a decision.
    const waiting = values.filter((item) => item.state === "blocked" && item.recovery).length;
    const badge = document.getElementById("inbox-count");
    if (badge) { badge.hidden = !waiting; badge.textContent = waiting + " waiting"; }
  }

  async function cancelOperation(host, id) {
    try {
      await api.post("/operations/cancel", { id });
    } catch (err) { renderDetailError(host, err); }
  }

  // decide records one operator decision bound to the approval request the operation advertises now.
  async function decide(id, tool, answer) {
    const view = await api.get("/operations/decisions");
    const blocked = (view.blocked || []).find((item) => item.operation === id);
    const action = blocked && (blocked.actions || []).find((item) => item.code === tool);
    if (!action || !action.request) throw new Error("operation no longer advertises this decision");
    return api.post("/operations/decision", { operation: id, tool, answer, request: action.request });
  }

  async function decideOperation(host, id, tool, answer) {
    try {
      await decide(id, tool, answer);
      await loadDetail(host, id);
    } catch (err) { renderDetailError(host, err); }
  }

  function renderDetailError(host, err) {
    const detail = host.lastElementChild;
    detail.hidden = false;
    detail.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
  }

  // tail: the live event tail of one operation, newest last, each event time-stamped here.
  function tail(id) {
    const rows = tails.get(id) || [];
    return overgo.table(["at", "state", "progress", "detail"], rows.map((event) => [
      new Date(event.at).toLocaleTimeString(), el("span", { text: event.state }), event.progress, el("span", { text: event.detail || "" })]));
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
    if (status.id && !status.local && !terminal(status.state) && status.state !== "blocked") {
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

    // The durable receipt: the run, its outputs and the traces behind it.
    const links = el("div", { class: "row" });
    if (status.run) links.appendChild(artifactLink(status.run, "run " + fmt.shortID(status.run)));
    for (const output of status.outputs || []) links.appendChild(artifactLink(output, "output " + fmt.shortID(output)));
    for (const document of projection.interactions || []) {
      const trace = document.value && document.value.trace;
      if (trace) links.appendChild(artifactLink(trace, "trace " + fmt.shortID(trace)));
    }

    const attempts = (projection.serving || []).map((document) => [document.value.attempt, el("span", { text: document.value.outcome }),
      (Number(document.value.measured_ns || 0) / 1e6).toFixed(2) + " ms", artifactLink(document.id)])
      .concat((status.attempts || []).map((attempt) => ["operation", el("span", { text: "recorded" }), "", artifactLink(attempt)]));
    const attemptTable = overgo.table(["attempt", "outcome", "duration", "evidence"], attempts);
    const stageTable = overgo.table(["stage", "state", "attempt", "failure"], (projection.stages || []).map((document) => [
      document.value.node, el("span", { text: document.value.state }), document.value.attempt, el("span", { text: document.value.failure || "" })]));
    const decisionTable = overgo.table(["answer", "tool", "arguments", "evidence"], (projection.decisions || []).map((document) => [
      document.value.answer, el("span", { text: document.value.tool }), (document.value.arguments || []).join(" "), artifactLink(document.id)]));

    detail.hidden = false;
    detail.replaceChildren(head, progress, stats, notices,
      el("div", { class: "section-title", text: "Live events" }), tail(projection.id),
      el("div", { class: "section-title", text: "Outputs and traces (the receipt)" }), links,
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
      const status = operations.get(id);
      // A local chip has no durable evidence; its detail is the chip's own facts.
      const projection = status && status.local ? { id, operation: status } : await api.get("/operations/evidence?id=" + encodeURIComponent(id), { signal: controller.signal });
      if (selected === id) renderDetail(host, projection);
    } catch (err) { if (!err || err.name !== "AbortError") renderDetailError(host, err); } finally {
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

  // note records one event in an operation's live tail, bounded to the last dozen.
  function note(status, detail) {
    const rows = tails.get(status.id) || [];
    rows.push({ at: Date.now(), state: status.state, progress: operationProgress(status), detail: detail || status.failure || "" });
    tails.set(status.id, rows.slice(-12));
  }

  // localOperation: a page reports work the runtime does not track (a model switch) as a chip.
  function localOperation(status) {
    status.local = true;
    operations.set(status.id, status);
    note(status, status.detail);
    if (shellHost) {
      renderStrip(shellHost);
      if (selected === status.id) renderDetail(shellHost, { id: status.id, operation: status });
    }
  }

  function start() {
    const host = document.getElementById("global-operation-shell");
    if (!host) return;
    shellHost = host;
    host.replaceChildren(el("div", { class: "operation-strip" }), el("div", { class: "operation-detail", hidden: true }));
    selected = routeOperation();
    renderStrip(host);
    if (selected) loadDetail(host, selected);
    overgo.runtimeEvents.subscribe((name, value) => {
      if (name === "operation.snapshot") {
        for (const [id, item] of operations) if (!item.local) operations.delete(id);
        for (const item of value || []) operations.set(item.id, item);
        renderStrip(host);
        if (!selected) {
          host.lastElementChild.hidden = true;
          host.lastElementChild.replaceChildren();
        }
      }
      if (name === "operation") {
        operations.set(value.status.id, value.status);
        note(value.status, value.stage && value.stage.node ? "stage " + value.stage.node + " " + (value.stage.state || "") : "");
        renderStrip(host);
        if (selected === value.status.id) loadDetail(host, selected);
      }
      if (name === "stream.error") renderDetailError(host, value);
    });
    window.addEventListener("popstate", () => {
      const id = routeOperation();
      if (id !== selected) selectOperation(host, id, true);
    });
    const key = document.getElementById("api-key");
    if (key) key.addEventListener("change", () => overgo.runtimeEvents.restart());
    const badge = document.getElementById("inbox-count");
    if (badge) badge.addEventListener("click", () => { location.hash = "inbox"; });
  }

  window.overgo.localOperation = localOperation;
  window.overgo.decideOperation = decide;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
