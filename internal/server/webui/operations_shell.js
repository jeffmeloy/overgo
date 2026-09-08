/* One compact activity entry; the existing runtime stream and durable receipts
   supply an on-demand list, decisions, cancellation and results. */
(function () {
  "use strict";

  const overgo = window.overgo;
  const { api, el, fmt } = overgo;
  const operations = new Map();
  const tails = new Map(); // operation id -> the last events the stream reported
  const dismissed = new Set();
  let selected = "";
  let detailRequest = null;
  let shellHost = null;
  let activityDialog, detailHost, listHost, streamHost, summaryButton;
  let connectionError = null;
  let activityOpener = null;
  let actionError = null;
  const pending = new Set();

  function terminal(state) { return state === "completed" || state === "cancelled" || state === "failed"; }

  function routeOperation() { return new URL(window.location.href).searchParams.get("operation") || ""; }

  function updateRoute(id, replace) {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set("operation", id);
    else url.searchParams.delete("operation");
    if (url.href === window.location.href) return;
    window.history[replace ? "replaceState" : "pushState"](null, "", url);
  }

  const artifactLink = window.overgo.artifactLink;

  function operationProgress(status) {
    const progress = status && status.progress || {};
    return progress.total == null ? String(progress.completed || 0) : progress.completed + " / " + progress.total;
  }

  function renderStrip(host) {
    listHost.hidden = !!selected;
    const values = [...operations.values()].filter((item) => item.id === selected || !dismissed.has(item.id) && (!terminal(item.state) || item.state === "failed"));
    const relevant = values.filter(item => !terminal(item.state) || item.state === 'failed');
    host.hidden = relevant.length === 0 && !connectionError;
    const active = values.filter((item) => !terminal(item.state)).length;
    const blocked = values.filter((item) => item.state === "blocked").length;
    const failed = values.filter((item) => item.state === "failed").length;
    const attentionLabel = (blocked + failed) + (blocked + failed === 1 ? ' needs attention' : ' need attention');
    summaryButton.textContent = connectionError ? 'Activity connection needs attention' : blocked || failed ? attentionLabel : active === 1 ? relevant[0].task + ' · ' + relevant[0].state : active ? active + ' tasks running' : 'Activity';
    const focused = listHost.contains(document.activeElement) && document.activeElement.dataset.operation;
    listHost.replaceChildren();
    for (const item of [...operations.values()].reverse()) {
      listHost.appendChild(el("button", {
        class: "operation-chip " + item.state + (selected === item.id ? " active" : ""),
        'data-operation': item.id,
        'aria-pressed': item.id === selected ? 'true' : 'false',
        text: item.task + ' · ' + item.state + (item.progress && item.progress.total != null ? " · " + operationProgress(item) : ""),
        title: item.task + " / " + item.id,
        onclick: () => selectOperation(host, item.id, false),
      }));
    }
    if (focused) [...listHost.children].find(button => button.dataset.operation === focused)?.focus();
    if (!listHost.children.length) listHost.appendChild(el('p', { class: 'note', text: 'No operations recorded in this session.' }));
    // The approvals count: every blocked operation waiting on a decision.
    const waiting = values.filter((item) => item.state === "blocked" && item.recovery).length;
    const badge = document.getElementById("inbox-count");
    if (badge) { badge.hidden = !waiting; badge.textContent = waiting + " waiting"; }
  }

  async function cancelOperation(host, id) {
    if (pending.has(id)) return;
    actionError = null;
    pending.add(id); detailHost.querySelectorAll('button').forEach(button => { if (button.textContent !== 'Close') button.disabled = true; });
    let failed = false;
    try {
      await api.post("/operations/cancel", { id });
    } catch (err) { failed = true; if (selected === id) renderDetailError(host, err, () => cancelOperation(host, id)); }
    finally { pending.delete(id); if (!failed && selected === id) loadDetail(host, id); }
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
    if (pending.has(id)) return;
    actionError = null;
    pending.add(id); detailHost.querySelectorAll('button').forEach(button => { if (button.textContent !== 'Close') button.disabled = true; });
    let failed = false;
    try {
      await decide(id, tool, answer);
    } catch (err) { failed = true; if (selected === id) renderDetailError(host, err, () => decideOperation(host, id, tool, answer)); }
    finally { pending.delete(id); if (!failed && selected === id) loadDetail(host, id); }
  }

  function renderDetailError(host, err, retry) {
    if (retry) actionError = { id: selected, err, retry };
    detailHost.hidden = false;
    detailHost.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)), el('button', { class: 'btn alt', text: retry ? 'Retry action' : 'Retry details', onclick: retry || (() => loadDetail(host, selected)) }));
  }

  function openActivity() {
    if (activityDialog.open) return;
    activityOpener = document.activeElement; activityDialog.showModal();
  }
  function closeActivity() {
    selected = ''; actionError = null; updateRoute('', true);
    if (detailRequest) detailRequest.abort();
    activityDialog.close(); detailHost.hidden = true; detailHost.replaceChildren(); renderStrip(shellHost);
    if (activityOpener && activityOpener.isConnected && activityOpener.getClientRects().length && !activityOpener.closest('[hidden]')) activityOpener.focus();
    else document.getElementById('settings-toggle').focus();
  }
  function renderConnection() {
    streamHost.replaceChildren();
    if (connectionError) streamHost.append(overgo.errorBanner(overgo.friendlyError(connectionError)),
      el('button', { class: 'btn alt', text: 'Retry connection', onclick: event => { event.currentTarget.disabled = true; overgo.runtimeEvents.restart(); } }),
      el('button', { class: 'link-button', text: 'Connection settings', onclick: () => { closeActivity(); document.getElementById('settings-toggle').click(); } }));
    renderStrip(shellHost);
  }

  // tail: the live event tail of one operation, newest last, each event time-stamped here.
  function tail(id) {
    const rows = tails.get(id) || [];
    return overgo.table(["at", "state", "progress", "detail"], rows.map((event) => [
      new Date(event.at).toLocaleTimeString(), el("span", { text: event.state }), event.progress, el("span", { text: event.detail || "" })]));
  }

  function renderDetail(host, projection) {
    const detail = detailHost;
    const focused = detail.contains(document.activeElement) && document.activeElement.dataset.action;
    const expanded = !!detail.querySelector('details[open]');
    const status = projection.operation || operations.get(projection.id) || {};
    if (status.id) operations.set(status.id, status);
    if (actionError && actionError.id === status.id) {
      if (!terminal(status.state)) { renderDetailError(host, actionError.err, actionError.retry); renderStrip(host); return; }
      actionError = null;
    }
    const summary = projection.summary || {};
    const usage = summary.usage || {};
    const resources = summary.resources || {};
    const head = el("div", { class: "operation-detail-head" },
      el("span", { class: "tag " + (status.state === "completed" ? "user_defined" : "control"), text: status.state || "durable" }),
      el("strong", { text: status.task || 'Operation' }),
      el("span", { class: "grow" }),
      el("button", { class: "link-button", 'data-action': 'back', text: "Back to activity", onclick: () => selectOperation(host, '', false) }));
    if (status.id && !status.local && !terminal(status.state) && status.state !== "blocked") {
      head.insertBefore(el("button", { class: "btn alt", 'data-action': 'cancel', text: "Cancel", onclick: () => cancelOperation(host, status.id) }), head.lastElementChild);
    }
    if (status.id && terminal(status.state)) head.insertBefore(el("button", { class: "btn alt", 'data-action': 'dismiss', text: "Dismiss", onclick: () => { dismissed.add(status.id); closeActivity(); } }), head.lastElementChild);

    const progress = el("progress", { class: "workflow-progress", value: (status.progress && status.progress.completed) || 0 });
    if (status.progress && status.progress.total != null) progress.max = status.progress.total;
    else progress.hidden = true;
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
        el("button", { class: "btn", 'data-action': 'grant-' + action.code, text: "Grant " + action.code, onclick: () => decideOperation(host, status.id, action.code, "grant") }),
        el("button", { class: "btn alt", 'data-action': 'decline-' + action.code, text: "Decline", 'aria-label': 'Decline ' + action.code, onclick: () => decideOperation(host, status.id, action.code, "decline") }));
      notices.append(el("div", { class: "section-title", text: "Recovery decision" }), actions);
    }

    // The durable receipt: the run, its outputs and the traces behind it.
    const links = el("div", { class: "row" }), evidenceLinks = el('div', { class: 'row' });
    if (status.run) evidenceLinks.appendChild(artifactLink(status.run, "run " + fmt.shortID(status.run)));
    for (const [index, output] of (status.outputs || []).entries()) links.appendChild(artifactLink(output, 'Download result' + (status.outputs.length > 1 ? ' ' + (index + 1) : '')));
    for (const document of projection.interactions || []) { const trace = document.value && document.value.trace; if (trace) evidenceLinks.appendChild(artifactLink(trace, "trace " + fmt.shortID(trace))); }

    const attempts = (projection.serving || []).map((document) => [document.value.attempt, el("span", { text: document.value.outcome }),
      (Number(document.value.measured_ns || 0) / 1e6).toFixed(2) + " ms", artifactLink(document.id)])
      .concat((status.attempts || []).map((attempt) => ["operation", el("span", { text: "recorded" }), "", artifactLink(attempt)]));
    const attemptTable = overgo.table(["attempt", "outcome", "duration", "evidence"], attempts);
    const stageTable = overgo.table(["stage", "state", "attempt", "failure"], (projection.stages || []).map((document) => [
      document.value.node, el("span", { text: document.value.state }), document.value.attempt, el("span", { text: document.value.failure || "" })]));
    const decisionTable = overgo.table(["answer", "tool", "arguments", "evidence"], (projection.decisions || []).map((document) => [
      document.value.answer, el("span", { text: document.value.tool }), (document.value.arguments || []).join(" "), artifactLink(document.id)]));

    detail.hidden = false;
    detail.replaceChildren(head, progress, notices,
      ...(links.children.length ? [el('div', { class: 'section-title', text: 'Results' }), links] : []),
      el('details', { open: expanded }, el('summary', { 'data-action': 'technical', text: 'Technical details' }), el('div', { class: 'mono', text: projection.id }), evidenceLinks, stats,
      el("div", { class: "section-title", text: "Live events" }), tail(projection.id),
      el("div", { class: "operation-evidence-columns" },
        el("div", {}, el("div", { class: "section-title", text: "Attempts" }), attemptTable),
        el("div", {}, el("div", { class: "section-title", text: "Stages" }), stageTable),
        el("div", {}, el("div", { class: "section-title", text: "Decisions" }), decisionTable))));
    if (pending.has(status.id)) detail.querySelectorAll('button').forEach(button => { if (button.textContent !== 'Close') button.disabled = true; });
    if (focused) ([...detail.querySelectorAll('button, summary')].find(button => button.dataset.action === focused) || head.lastElementChild).focus({ preventScroll: true });
    renderStrip(host);
  }

  async function loadDetail(host, id, focus) {
    if (detailRequest) detailRequest.abort();
    const controller = new AbortController();
    detailRequest = controller;
    try {
      const status = operations.get(id);
      // A local chip has no durable evidence; its detail is the chip's own facts.
      const projection = status && status.local ? { id, operation: status } : await api.get("/operations/evidence?id=" + encodeURIComponent(id), { signal: controller.signal });
      if (selected === id && detailRequest === controller && !controller.signal.aborted) {
        renderDetail(host, projection);
        if (focus && activityDialog.open) { detailHost.focus(); activityDialog.scrollTop = 0; }
      }
    } catch (err) { if (selected === id && detailRequest === controller && !controller.signal.aborted && (!err || err.name !== "AbortError")) renderDetailError(host, err); } finally { if (detailRequest === controller) detailRequest = null; }
  }

  function selectOperation(host, id, replace) {
    const previous = selected;
    if (selected !== id) actionError = null;
    selected = id;
    updateRoute(id, replace);
    renderStrip(host);
    openActivity();
    if (!id) { if (detailRequest) detailRequest.abort(); detailHost.hidden = true; detailHost.replaceChildren(); ([...listHost.children].find(button => button.dataset.operation === previous) || listHost.querySelector('button'))?.focus(); return; }
    detailHost.hidden = false; detailHost.replaceChildren(el('p', { class: 'note', role: 'status', text: 'Loading operation…' }));
    loadDetail(host, id, true);
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
    if (!terminal(status.state)) dismissed.delete(status.id);
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
    listHost = el('div', { class: 'activity-list', 'aria-label': 'Operations and results' });
    detailHost = el('section', { class: 'operation-detail', hidden: true, tabindex: '-1', 'aria-label': 'Operation details' });
    streamHost = el('div');
    activityDialog = el('dialog', { class: 'surface-dialog activity-dialog', 'aria-label': 'Activity' },
      el('div', { class: 'dialog-heading' }, el('h2', { text: 'Activity' }), el('button', { class: 'btn alt', text: 'Close', 'aria-label': 'Close activity', onclick: closeActivity })), streamHost, listHost, detailHost);
    activityDialog.addEventListener('cancel', event => { event.preventDefault(); closeActivity(); });
    document.body.appendChild(activityDialog);
    summaryButton = el('button', { class: 'link-button activity-summary', 'aria-haspopup': 'dialog', onclick: openActivity });
    host.replaceChildren(el('div', { class: 'operation-strip', 'aria-live': 'polite' }, summaryButton));
    selected = routeOperation();
    renderStrip(host);
    if (selected) { openActivity(); loadDetail(host, selected); }
    overgo.runtimeEvents.subscribe((name, value) => {
      if (name === "operation.snapshot") {
        for (const [id, item] of operations) if (!item.local) operations.delete(id);
        for (const item of value || []) operations.set(item.id, item);
        renderStrip(host);
        if (selected) loadDetail(host, selected);
      }
      if (name === "operation") {
        operations.set(value.status.id, value.status);
        note(value.status, value.stage && value.stage.node ? "stage " + value.stage.node + " " + (value.stage.state || "") : "");
        renderStrip(host);
        if (selected === value.status.id) loadDetail(host, selected);
      }
      if (name === 'stream.ready' || name === 'stream.idle') { connectionError = null; renderConnection(); }
      if (name === "stream.error") { connectionError = value; renderConnection(); }
    });
    window.addEventListener("popstate", () => { const id = routeOperation(); if (id !== selected) { if (id) selectOperation(host, id, true); else closeActivity(); } });
    const badge = document.getElementById("inbox-count");
    if (badge) badge.addEventListener("click", () => { location.hash = "inbox"; });
  }

  window.overgo.localOperation = localOperation;
  window.overgo.decideOperation = decide;
  window.overgo.showOperation = id => selectOperation(shellHost, id || '', false);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
