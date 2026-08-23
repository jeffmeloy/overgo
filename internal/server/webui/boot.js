/* overgo_gui boot: the thin-client core. Defines window.overgo (fetch helpers
   with bearer injection, a DOM helper, a tab registry) and wires the shell.
   Analysis modules self-register a tab; nothing here holds state a re-fetch
   can't rebuild. Classic deferred script so modules registered after it (also
   deferred) are all present by DOMContentLoaded. */
(function () {
  "use strict";
  const KEY_STORAGE = "overgo.apiKey";

  function getKey() { return localStorage.getItem(KEY_STORAGE) || ""; }
  function setKey(value) {
    if (value) localStorage.setItem(KEY_STORAGE, value);
    else localStorage.removeItem(KEY_STORAGE);
  }

  function authHeaders(extra) {
    const headers = Object.assign({}, extra || {});
    const key = getKey();
    if (key) headers["Authorization"] = "Bearer " + key;
    return headers;
  }

  // Parse a JSON body; surface the server's error message on failure.
  async function readJSON(response) {
    const text = await response.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch (_) { /* non-JSON */ }
    if (!response.ok) {
      const message = (body && body.error && (body.error.message || body.error)) ||
        (body && body.message) || text || ("HTTP " + response.status);
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return body;
  }

  const api = {
    async get(path, opts) {
      return readJSON(await fetch(path, { headers: authHeaders(), signal: opts && opts.signal }));
    },
    async post(path, body, opts) {
      return readJSON(await fetch(path, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body),
        signal: opts && opts.signal,
      }));
    },
    async stream(path, body, opts) {
      const response = await fetch(path, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body),
        signal: opts && opts.signal,
      });
      if (!response.ok) await readJSON(response);
      return response;
    },
    async events(path, handler, opts) {
	  const separator = "\n\n";
	  const eventPrefix = "event: ";
	  const dataPrefix = "data: ";
      const response = await fetch(path, { headers: authHeaders(), signal: opts && opts.signal });
      if (!response.ok) await readJSON(response);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffered = "";
      for (;;) {
        const chunk = await reader.read();
        buffered += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
        let boundary;
		while ((boundary = buffered.indexOf(separator)) >= 0) {
          const block = buffered.slice(0, boundary);
		  buffered = buffered.slice(boundary + separator.length);
          let event = "message";
          const data = [];
          for (const line of block.split("\n")) {
			if (line.startsWith(eventPrefix)) event = line.slice(eventPrefix.length);
			else if (line.startsWith(dataPrefix)) data.push(line.slice(dataPrefix.length));
          }
          if (data.length) handler(event, JSON.parse(data.join("\n")));
        }
        if (chunk.done) return;
      }
    },
  };

  // /analyze/model is fetched by the shell (capability gating), the Model tab,
  // and the lens (vocab size). Cache the in-flight/last promise so a page load
  // hits it once; a rejection clears the cache so a retry after the key is set
  // refetches, and a key change invalidates it explicitly.
  let modelPromise = null;
  function modelInfo() {
    if (!modelPromise) {
      modelPromise = api.get("/analyze/model").catch((err) => { modelPromise = null; throw err; });
    }
    return modelPromise;
  }
  function invalidateModel() { modelPromise = null; }

  // Minimal hyperscript: el("div", {class:"x"}, child, child...).
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [name, value] of Object.entries(attrs)) {
        if (value == null || value === false) continue;
        if (name === "class") node.className = value;
        else if (name === "text") node.textContent = value;
        else if (name === "html") node.innerHTML = value;
        else if (name.startsWith("on") && typeof value === "function") node.addEventListener(name.slice(2), value);
        else node.setAttribute(name, value);
      }
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function errorBanner(message) {
    return el("div", { class: "err-banner", text: message });
  }

  // friendlyError: one place that turns a 401 into the actionable hint every tab
  // should show, so auth failures read the same everywhere instead of leaking
  // the raw "missing or invalid bearer token".
  function friendlyError(err) {
    if (err && err.status === 401) return "API key required — enter it in the top bar.";
    return String((err && err.message) || err);
  }

  // Number formatting helpers (grouping, byte sizes, compact counts).
  function grouped(n) { return Number(n).toLocaleString("en-US"); }
  function bytes(n) {
    n = Number(n);
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return (i === 0 ? n : n.toFixed(2)) + " " + units[i];
  }
  function compact(n) {
    n = Number(n);
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(n);
  }
  function shortID(id) {
    id = String(id || "");
    const separator = id.indexOf(":");
    const hash = separator < 0 ? id : id.slice(id.lastIndexOf(":") + 1);
    return (separator < 0 ? "" : id.slice(0, separator) + ":") + hash.slice(0, 10);
  }

  // displayToken: make an empty / whitespace-only / multiline token piece
  // visible without altering the underlying text. Shared by every tab that shows
  // token strings (logit lens, hidden states, attention).
  function displayToken(text) {
    if (text === "" || text == null) return "∅";
    if (/^\s+$/.test(text)) return "␠".repeat(text.length);
    return text.replace(/\n/g, "⏎");
  }

  // runner: manage an exclusive, cancelable async action bound to a run button
  // and a cancel button. run(task) ignores re-entrant calls, disables run and
  // reveals cancel while task(signal) is in flight, and dispatches an abort to
  // onCancel and any other failure to onError. Factored out of the three tabs
  // that drive a real forward pass so the abort semantics live in one place.
  function runner(runButton, cancelButton, handlers) {
    handlers = handlers || {};
    let controller = null;
    cancelButton.addEventListener("click", () => controller && controller.abort());
    return async function run(task) {
      if (controller) return; // a run is already in flight
      runButton.disabled = true;
      cancelButton.style.display = "";
      controller = new AbortController();
      try {
        await task(controller.signal);
      } catch (err) {
        if (err && err.name === "AbortError") { if (handlers.onCancel) handlers.onCancel(); }
        else if (handlers.onError) handlers.onError(err);
      } finally {
        runButton.disabled = false;
        cancelButton.style.display = "none";
        controller = null;
      }
    };
  }

  function poller(task, interval) {
    let active = false;
    let controller = null;
    let timer = null;

    function cancel() {
      if (timer != null) clearTimeout(timer);
      timer = null;
      if (controller) controller.abort();
    }
    function schedule() {
      if (active && !document.hidden) timer = setTimeout(tick, interval);
    }
    async function tick() {
      if (!active || document.hidden || controller) return;
      const current = new AbortController();
      controller = current;
      try { await task(current.signal); }
      catch (err) {
        if (!err || err.name !== "AbortError") throw err;
      } finally {
        if (controller === current) controller = null;
        schedule();
      }
    }
    function start() {
      if (active) return;
      active = true;
      if (!document.hidden) tick();
    }
    function stop() {
      active = false;
      cancel();
    }
    document.addEventListener("visibilitychange", () => {
      cancel();
      if (active && !document.hidden) tick();
    });
    return { start, stop };
  }

  function stat(label, value, unit) {
    return el("div", { class: "stat" },
      el("div", { class: "k", text: label }),
      el("div", { class: "v" }, String(value), unit ? el("small", { text: " " + unit }) : null));
  }

  const tabs = [];
  function registerTab(tab) { tabs.push(tab); }

  window.overgo = {
    api, el, clear, errorBanner, friendlyError, registerTab,
    getKey, setKey, modelInfo, invalidateModel,
    displayToken, runner, poller, stat,
    fmt: { grouped, bytes, compact, shortID },
  };

  // ---- shell wiring (runs after all deferred module scripts registered) ----
  // Two-level nav: top-level SECTIONS, each holding one or more tabs. A tab
  // declares `section`; those without one fall into "workbench".
  const SECTION_ORDER = [
    { id: "inference", label: "Inference" },
    { id: "datasets", label: "Datasets" },
    { id: "training", label: "Training" },
    { id: "workbench", label: "Workbench" },
  ];
  let activeSection = null;
  const sectionButtons = [];

  function tabSection(tab) { return tab.section || "workbench"; }
  function sectionsPresent() {
    return SECTION_ORDER.filter((s) => tabs.some((t) => tabSection(t) === s.id));
  }

  function syncSectionUI() {
    for (const sb of sectionButtons) sb.button.classList.toggle("active", sb.id === activeSection);
    for (const tab of tabs) tab.button.style.display = tabSection(tab) === activeSection ? "" : "none";
  }

  function activate(id) {
    const tab = tabs.find((t) => t.id === id);
    if (!tab) return;
    activeSection = tabSection(tab);
    for (const t of tabs) {
      const on = t.id === id;
      const wasActive = t.panel.classList.contains("active");
      if (!on && wasActive && t.onDeactivate) t.onDeactivate();
      t.button.classList.toggle("active", on);
      t.panel.classList.toggle("active", on);
      if (on && !t.mounted) { t.mounted = true; safeMount(t); }
      if (on && t.onActivate) t.onActivate();
    }
    syncSectionUI();
    if (location.hash.slice(1) !== id) history.replaceState(null, "", "#" + id);
  }

  // selectSection: switch to a section, keeping the current tab if it already
  // belongs there, otherwise activating the section's first (supported) tab.
  function selectSection(id) {
    const current = tabs.find((t) => t.button.classList.contains("active"));
    if (current && tabSection(current) === id) { activeSection = id; syncSectionUI(); return; }
    const first = tabs.find((t) => tabSection(t) === id && tabSupported(t)) ||
      tabs.find((t) => tabSection(t) === id);
    if (first) activate(first.id);
  }

  function safeMount(tab) {
    try {
      const result = tab.mount(tab.panel, window.overgo);
      if (result && typeof result.catch === "function") {
        result.catch((err) => renderMountError(tab, err));
      }
    } catch (err) { renderMountError(tab, err); }
  }
  function renderMountError(tab, err) {
    clear(tab.panel);
    tab.panel.appendChild(errorBanner(String(err && err.message || err)));
  }

  async function refreshStatus() {
    const statusPill = document.getElementById("status-pill");
    const modelPill = document.getElementById("model-pill");
    try {
      const health = await api.get("/health");
      statusPill.textContent = "online";
      statusPill.className = "pill ok";
      if (health && health.model) modelPill.textContent = health.model;
    } catch (err) {
      statusPill.textContent = "offline";
      statusPill.className = "pill err";
    }
  }

  // ---- capability gating ----
  // A tab may declare `requires: "<key>"`; if /analyze/model's analysis block
  // reports that capability false, the tab is disabled rather than allowed to
  // fail. Fail-open: if capabilities can't be fetched (offline / key required),
  // every tab stays enabled and errors surface per-request instead.
  let capabilities = null;
  let authNoticeEl = null;

  // When /analyze/model answers 401 the whole analysis surface is locked behind
  // the key; surface one banner + highlight the field instead of letting each
  // tab fail on its own with a raw bearer-token error.
  function showAuthNotice(show) {
    if (authNoticeEl) authNoticeEl.style.display = show ? "" : "none";
    const key = document.getElementById("api-key");
    if (key) key.classList.toggle("needs-key", show);
  }

  function tabSupported(tab) {
    if (!tab.requires || !capabilities) return true;
    return capabilities[tab.requires] !== false;
  }

  function applyCapabilities() {
    for (const tab of tabs) {
      const ok = tabSupported(tab);
      tab.button.disabled = !ok;
      tab.button.classList.toggle("disabled-tab", !ok);
      tab.button.title = ok ? "" : "Not available for this model";
    }
    const active = tabs.find((t) => t.button.classList.contains("active"));
    if (active && !tabSupported(active)) {
      const firstOk = tabs.find(tabSupported);
      if (firstOk) activate(firstOk.id);
    }
  }

  async function refreshCapabilities() {
    try {
      const model = await modelInfo();
      capabilities = model.analysis || {};
      showAuthNotice(false);
    } catch (err) {
      capabilities = null; // fail-open
      showAuthNotice(err && err.status === 401);
    }
    applyCapabilities();
  }

  function initShell() {
    const sectionBar = document.getElementById("sections");
    const tabBar = document.getElementById("tabs");
    const panels = document.getElementById("panels");
    for (const section of sectionsPresent()) {
      const button = el("button", { class: "section", onclick: () => selectSection(section.id) }, section.label);
      sectionBar.appendChild(button);
      sectionButtons.push({ id: section.id, button: button });
    }
    for (const tab of tabs) {
      tab.button = el("button", { class: "tab", onclick: () => activate(tab.id) }, tab.label);
      tab.panel = el("div", { class: "panel", id: "panel-" + tab.id });
      tabBar.appendChild(tab.button);
      panels.appendChild(tab.panel);
    }
    authNoticeEl = el("div", { class: "auth-banner", style: "display:none" },
      "This server requires an API key — enter it in the field at the top right to load analysis and chat.");
    document.querySelector(".wrap").insertBefore(authNoticeEl, panels);

    const keyInput = document.getElementById("api-key");
    keyInput.value = getKey();
    keyInput.addEventListener("change", () => {
      setKey(keyInput.value.trim());
      invalidateModel(); // the cached model was fetched under the old key
      // Re-mount the active tab so its data reloads under the new key.
      for (const tab of tabs) {
        if (tab.onDeactivate) tab.onDeactivate();
        tab.mounted = false;
      }
      const current = location.hash.slice(1) || (tabs[0] && tabs[0].id);
      if (current) activate(current);
      refreshStatus();
      refreshCapabilities();
    });
    window.addEventListener("hashchange", () => {
      const id = location.hash.slice(1);
      if (id && tabs.some((t) => t.id === id)) activate(id);
    });
    const start = location.hash.slice(1);
    activate(tabs.some((t) => t.id === start) ? start : (tabs[0] && tabs[0].id));
    refreshStatus();
    refreshCapabilities();
    // Re-probe health so a server that drops (or comes back) is reflected in the
    // status pill instead of showing a stale "online" until the next key change.
    setInterval(refreshStatus, 10000);
  }

  // boot.js is deferred, so it runs while readyState is "interactive" — before
  // the later deferred module scripts have registered their tabs. DOMContentLoaded
  // fires only after all deferred scripts run, so defer initShell to it unless the
  // document is already fully loaded.
  if (document.readyState === "complete") {
    initShell();
  } else {
    document.addEventListener("DOMContentLoaded", initShell);
  }
})();
