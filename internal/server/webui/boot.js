/* overgo_gui boot: the thin-client core. Defines window.overgo (fetch helpers
   with bearer injection, a DOM helper, a tab registry), loads the libraries
   and the manifest's modules, and wires the shell; nothing here holds state
   a re-fetch cannot rebuild. */
(function () {
  "use strict";
  const KEY_STORAGE = "overgo.apiKey";
  const MODEL_STORAGE = "overgo.model"; // the last model this browser chose to serve
  const errors = []; // every window error since boot; the browser lane asserts none
  window.addEventListener("error", (event) => errors.push(String(event.message)));
  window.addEventListener("unhandledrejection", (event) => errors.push(String(event.reason)));

  // The key lives in memory for the page session; browser storage is opt-in via the
  // "remember" control so a shared machine never keeps a key the operator did not ask it to keep.
  let sessionKey = "";
  try { sessionKey = localStorage.getItem(KEY_STORAGE) || ""; } catch (_) { /* storage unavailable */ }
  const keyWasPersisted = sessionKey !== "";

  function getKey() { return sessionKey; }
  function setKey(value, remember) {
    sessionKey = value || "";
    try {
      if (remember && sessionKey) localStorage.setItem(KEY_STORAGE, sessionKey);
      else localStorage.removeItem(KEY_STORAGE);
    } catch (_) { /* storage unavailable; the in-memory key still works */ }
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
      const response = await fetch(path, { headers: authHeaders(), signal: opts && opts.signal });
      if (opts && opts.onHeaders) opts.onHeaders(response.headers);
      return readJSON(response);
    },
    async post(path, body, opts) {
      return readJSON(await fetch(path, { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(body), signal: opts && opts.signal }));
    },
    // upload: a file's bytes under their own media type (opts.mediaType sends the body raw); the server answers the stored artifact.
    async upload(path, file, opts) { return readJSON(await this.stream(path, file, Object.assign({ mediaType: file.type }, opts))); },
    async stream(path, body, opts) {
      const method = (opts && opts.method) || "POST", raw = opts && opts.mediaType;
      const response = await fetch(path, { method, headers: authHeaders(method === "POST" ? { "Content-Type": raw || "application/json" } : {}), body: method !== "POST" ? undefined : raw ? body : JSON.stringify(body), signal: opts && opts.signal });
      if (!response.ok) await readJSON(response);
      return response;
    },
    async events(path, handler, opts) {
      const response = await fetch(path, { headers: authHeaders(), signal: opts && opts.signal });
      if (!response.ok) await readJSON(response);
      for await (const { event, data } of sseEvents(response)) handler(event, data);
    },
    // blob: an artifact's bytes (a media output taken back as input).
    async blob(path, opts) { return (await this.stream(path, null, Object.assign({ method: "GET" }, opts))).blob(); },
  };

  // sseEvents: the one reader of a server-sent event stream; yields each
  // block's event name and parsed data (the stream adapters and api.events).
  async function* sseEvents(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    for (;;) {
      const chunk = await reader.read();
      buffered += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
      let boundary;
      while ((boundary = buffered.indexOf("\n\n")) >= 0) {
        const block = buffered.slice(0, boundary);
        buffered = buffered.slice(boundary + 2);
        let event = "message";
        const data = [];
        for (const line of block.split("\n")) { if (line.startsWith("event: ")) event = line.slice(7); else if (line.startsWith("data: ")) data.push(line.slice(6)); }
        if (!data.length) continue;
        const joined = data.join("\n");
        if (joined === "[DONE]") { yield { event: "done", data: null }; continue; }
        let parsed;
        try { parsed = JSON.parse(joined); } catch (_) { continue; }
        yield { event, data: parsed };
      }
      if (chunk.done) return;
    }
  }

  // /analyze/model serves the Model tab and the lens; one cached promise per
  // page load, cleared on rejection and on a key change.
  let modelPromise = null;
  function modelInfo() { return modelPromise || (modelPromise = api.get("/analyze/model").catch((err) => { modelPromise = null; throw err; })); }
  function invalidateModel() { modelPromise = null; }

  // Minimal hyperscript: el("div", {class:"x"}, child, child...).
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [name, value] of Object.entries(attrs)) {
        if (value == null || value === false) continue;
        if (name === "class") node.className = value;
        else if (name === "text") node.textContent = value;
        else if (name.startsWith("on") && typeof value === "function") node.addEventListener(name.slice(2), value);
        else if ((name === "src" || name === "href") && String(value).startsWith("/artifacts/content?")) artifactResource(node, name, value);
        else node.setAttribute(name, value);
      }
    }
    for (const child of children.flat()) { if (child == null || child === false) continue; node.appendChild(typeof child === "string" ? document.createTextNode(child) : child); }
    return node;
  }

  // Native media and links cannot send a bearer header. Fetch protected bytes
  // through the shared client and release their object URLs when removed.
  const artifactURLs = new Map();
  new MutationObserver(() => {
    for (const [node, url] of artifactURLs) {
      if (!node.isConnected) { URL.revokeObjectURL(url); artifactURLs.delete(node); }
    }
  }).observe(document.documentElement, { childList: true, subtree: true });
  function artifactResource(node, attribute, path) {
    const load = async () => {
      try {
        const body = await api.blob(path);
        if (!node.isConnected) return;
        if (artifactURLs.has(node)) URL.revokeObjectURL(artifactURLs.get(node));
        const url = URL.createObjectURL(body);
        artifactURLs.set(node, url);
        if (attribute === "src") node.src = url;
        else {
          const download = el("a", { href: url, download: node.getAttribute("download") || "artifact" });
          download.click();
        }
      } catch (err) { node.replaceWith(errorBanner(friendlyError(err))); }
    };
    if (attribute === "href") {
      node.setAttribute("href", path);
      node.addEventListener("click", (event) => { event.preventDefault(); load(); });
    } else load();
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function errorBanner(message) { return el("div", { class: "err-banner", role: "alert", text: message }); }

  // friendlyError: a 401 becomes the same actionable hint on every tab.
  function friendlyError(err) { return err && err.status === 401 ? "API key required — open Settings and enter it under Connection." : String((err && err.message) || err); }

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

  // displayToken: an empty, whitespace or multiline token piece made visible.
  function displayToken(text) { if (text === "" || text == null) return "∅"; if (/^\s+$/.test(text)) return "␠".repeat(text.length); return text.replace(/\n/g, "⏎"); }

  // runner: one exclusive, cancelable async action bound to a run and a cancel
  // button; an abort goes to onCancel, any other failure to onError.
  // The shell's timers, named in one place: a library field's focus retry, the swap button's loading
  // tick, the offline probe, and the status re-probe.
  const focusRetryMS = 50, loadingTickMS = 1000, offlineProbeMS = 4000, statusRefreshMS = 10000;
  function runner(runButton, cancelButton, handlers) {
    handlers = handlers || {};
    let controller = null;
    cancelButton.addEventListener("click", () => controller && controller.abort());
    return async function run(task) {
      if (controller) return; // a run is already in flight
      runButton.disabled = true;
      cancelButton.hidden = false;
      controller = new AbortController();
      try {
        await task(controller.signal);
      } catch (err) {
        if (err && err.name === "AbortError") { if (handlers.onCancel) handlers.onCancel(); }
        else if (handlers.onError) handlers.onError(err);
      } finally { runButton.disabled = false; cancelButton.hidden = true; controller = null; }
    };
  }

  function poller(task, interval) {
    let active = false;
    let controller = null;
    let timer = null;

    function cancel() { if (timer != null) clearTimeout(timer); timer = null; if (controller) controller.abort(); }
    function schedule() { if (active && !document.hidden) timer = setTimeout(tick, interval); }
    async function tick() {
      if (!active || document.hidden || controller) return;
      const current = new AbortController();
      controller = current;
      try { await task(current.signal); }
      catch (err) { if (!err || err.name !== "AbortError") throw err; } finally { if (controller === current) controller = null; schedule(); }
    }
    function start() { if (active) return; active = true; if (!document.hidden) tick(); }
    function stop() { active = false; cancel(); }
    document.addEventListener("visibilitychange", () => { cancel(); if (active && !document.hidden) tick(); });
    return { start, stop };
  }

  function stat(label, value, unit) {
    return el("div", { class: "stat" }, el("div", { class: "k", text: label }), el("div", { class: "v" }, String(value), unit ? el("small", { text: " " + unit }) : null));
  }

  // fold: a collapsible section, closed unless open is passed.
  function fold(title, open, ...children) {
    const details = el("details", { class: "fold" }, el("summary", { class: "section-title", text: title }), ...children);
    if (open) details.setAttribute("open", "");
    return details;
  }

  const tabs = [];
  function registerTab(tab) { tabs.push(tab); }

  // artifactLink: the one link to a stored artifact (content, or the gallery entry).
  const galleryRoute = "/artifacts?id=", contentRoute = "/artifacts/content?id=";
  function artifactLink(id, label, gallery) {
    const route = gallery ? galleryRoute : contentRoute;
    return el("a", { class: "mono", href: route + encodeURIComponent(id), target: "_blank", rel: "noopener", text: label || shortID(id) });
  }

  // headerRow, tableRow, table: a header row from labels, a row of cells (a
  // node, or text shown mono after the first column), a grid table from both.
  function headerRow(labels) { return el("tr", {}, ...labels.map((text) => el("th", { text }))); }
  // reporter: a host's error reporter, the shape every module spells as showError.
  function reporter(host) { return (err) => host.replaceChildren(errorBanner(friendlyError(err))); }
  function tableRow(cells, attrs) {
    return el("tr", attrs || {}, ...cells.map((cell, index) => cell instanceof Node ? el("td", {}, cell) : el("td", { class: index ? "mono" : "", text: String(cell == null ? "" : cell) })));
  }
  function table(headers, rows, cls) { return el("table", { class: "grid " + (cls || "") }, headerRow(headers), ...rows.map((row) => tableRow(row))); }

  // evidenceLine: a catalog entry's committed evidence, shown by the header and the picker alike.
  function evidenceLine(item) {
    const facts = [];
    const benchmark = item.benchmark || {};
    if (benchmark.prompt_tokens_per_second_p50) facts.push(Math.round(benchmark.prompt_tokens_per_second_p50) + " prompt tok/s");
    if (benchmark.decode_tokens_per_second_p50) facts.push(Math.round(benchmark.decode_tokens_per_second_p50) + " decode tok/s");
    for (const entry of item.evals || []) if (entry.metrics && typeof entry.metrics.accuracy === "number") facts.push((entry.suite || "").replace(/^store\//, "") + " " + entry.metrics.accuracy.toFixed(2) + (entry.reproducible === false ? " (hosted)" : ""));
    return facts.join(" · ");
  }

  // The front page: the workbench folds to a rail; the Workbench control or any other tab unfolds it.
  function setFront(on) {
    document.querySelector(".shell").classList.toggle("front", on);
    document.getElementById("workbench-toggle").setAttribute("aria-expanded", String(!on));
  }
  const phoneNavigation = matchMedia("(max-width: 860px)");
  function closeNavigation() {
    const nav = document.getElementById("navigation");
    const toggle = document.getElementById("navigation-toggle");
    const wasOpen = toggle.getAttribute("aria-expanded") === "true";
    document.querySelector(".shell").classList.remove("navigation-open");
    toggle.setAttribute("aria-expanded", "false");
    document.getElementById("navigation-backdrop").hidden = true;
    document.querySelector(".content").inert = false;
    nav.inert = phoneNavigation.matches;
    nav.removeAttribute("role"); nav.removeAttribute("aria-modal");
    if (wasOpen) toggle.focus();
  }
  function openNavigation() {
    const nav = document.getElementById("navigation");
    nav.inert = false;
    nav.setAttribute("role", "dialog"); nav.setAttribute("aria-modal", "true");
    document.querySelector(".shell").classList.add("navigation-open");
    document.getElementById("navigation-toggle").setAttribute("aria-expanded", "true");
    document.getElementById("navigation-backdrop").hidden = false;
    document.querySelector(".content").inert = true;
    document.getElementById("navigation-close").focus();
  }
  document.getElementById("navigation-toggle").addEventListener("click", openNavigation);
  document.getElementById("navigation-close").addEventListener("click", closeNavigation);
  document.getElementById("navigation-backdrop").addEventListener("click", closeNavigation);
  phoneNavigation.addEventListener("change", closeNavigation);
  closeNavigation();
  document.getElementById("navigation").addEventListener("keydown", (event) => {
    if (!phoneNavigation.matches) return;
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); closeNavigation(); }
    if (event.key !== "Tab") return;
    const controls = [...event.currentTarget.querySelectorAll("button:not(:disabled), a[href], input")].filter((node) => node.getClientRects().length);
    const first = controls[0], last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
  const settingsDialog = document.getElementById("settings-dialog");
  document.getElementById("settings-toggle").addEventListener("click", (event) => { event.currentTarget.focus(); settingsDialog.showModal(); });
  settingsDialog.querySelector("[data-close-dialog]").addEventListener("click", () => settingsDialog.close());
  document.getElementById("new-chat").addEventListener("click", () => openConversation(null));
  // Follow the browser's visible height when its on-screen keyboard resizes only the visual viewport.
  if (window.visualViewport) {
    const resize = () => { if (window.visualViewport.scale === 1) document.documentElement.style.setProperty("--viewport-height", window.visualViewport.height + "px"); };
    window.visualViewport.addEventListener("resize", resize); resize();
    window.addEventListener("resize", resize);
  }
  let servedEntry = null;
  let modelSwitchPending = false;
  let modelSwitchBlocked = false;
  function modelSwitching() { return modelSwitchBlocked; }
  function setModelSwitchBlocked(value) { modelSwitchBlocked = value; document.dispatchEvent(new Event("overgo-model-switch")); }
  function servedModel() { return servedEntry; }

  // ---- conversations: the server lists stored-response chains; the rail opens, renames or archives one ----
  let selectedConversation = null;
  try { selectedConversation = JSON.parse(sessionStorage.getItem("overgo.conversation") || "null"); } catch (_) { /* storage unavailable */ }
  function conversation() { return selectedConversation; }
  function rememberConversation(item) {
    selectedConversation = item;
    try { sessionStorage.setItem("overgo.conversation", JSON.stringify(item)); } catch (_) { /* storage unavailable */ }
  }
  function openConversation(item) {
    setFront(true);
    rememberConversation(item);
    for (const tab of tabs) { if (tab.id === "chat") tab.mounted = false; }
    activate("chat");
  }
  async function refreshConversations() {
    const host = document.getElementById("conversation-list");
    if (!host) return;
    let listing;
    try { listing = await api.get("/interactions"); } catch (_) { clear(host); return; }
    const fresh = el("button", { class: "tab", text: "+ new conversation", onclick: () => openConversation(null) });
    clear(host);
    host.appendChild(fresh);
    for (const item of (listing.conversations || []).filter((entry) => !entry.archived)) {
      const open = el("button", { class: "tab" + (selectedConversation && selectedConversation.root === item.root ? " active" : ""),
        text: item.title, title: item.turns + " turn(s)", onclick: () => openConversation(item) });
      const label = async (patch) => {
        await api.post("/interactions/label", { root: item.root, title: item.title, archived: false, ...patch });
        if (patch.archived && selectedConversation && selectedConversation.root === item.root) openConversation(null);
        refreshConversations();
      };
      // Rename in place: the title becomes a field; Enter saves, Escape restores the rail.
      const rename = el("button", { class: "link-button", text: "rename", "aria-label": "rename conversation", onclick: () => {
        const field = el("input", { class: "text", value: item.title, "aria-label": "conversation title" });
        field.addEventListener("keydown", (event) => { if (event.key === "Enter") label({ title: field.value.trim() || item.title }); else if (event.key === "Escape") refreshConversations(); });
        open.replaceWith(field); field.focus(); field.select();
      } });
      const archive = el("button", { class: "link-button", text: "archive", "aria-label": "archive conversation", onclick: () => label({ archived: true }) });
      host.appendChild(el("div", { class: "conversation" }, open, el("div", { class: "row" }, rename, archive)));
    }
  }

  window.overgo = {
    api, el, clear, errorBanner, friendlyError, registerTab, artifactLink, headerRow, tableRow, table, evidenceLine, servedModel, modelSwitching,
    conversation, rememberConversation, openConversation, refreshConversations, sseEvents, errors, embed, analysisSurface, reporter,
    getKey, setKey, modelInfo, invalidateModel,
    displayToken, runner, poller, stat, fold,
    fmt: { grouped, bytes, compact, shortID },
  };

  // ---- shell wiring: the server manifest owns navigation order, labels and refusal; modules register only ----
  let workspaceManifest = null;
  let activeSection = null;
  const sectionButtons = [];

  function tabSection(tab) { return tab.section; }
  function sectionsPresent() { return workspaceManifest.sections.filter((s) => tabs.some((t) => tabSection(t) === s.id)); }

  // Sidebar navigation shows every section at once; the active section header highlights the active tab group.
  function syncSectionUI() { for (const sb of sectionButtons) sb.button.classList.toggle("active", sb.id === activeSection); }

  // remountActive: the active tab reloads under a new key, a newly served model or a
  // changed store, from the capability document re-read for it.
  async function remountActive(manifest) {
    const next = manifest || await api.get("/workspace/manifest");
    workspaceManifest = next;
    capabilityDocument = next.model || null;
    for (const tab of tabs) {
      if (tab.onDeactivate) tab.onDeactivate();
      tab.mounted = false;
      // The newly served model's refusals replace the last one's, so the nav shows what works now.
      const declared = (workspaceManifest.tabs || []).find((declaration) => declaration.id === tab.id);
      if (declared) { tab.enabled = declared.enabled; tab.refusal = declared.refusal; }
    }
    applyCapabilities();
    const current = location.hash.slice(1) || (tabs[0] && tabs[0].id);
    if (current && (capabilityDocument || tabs.some((t) => t.id === location.hash.slice(1)))) {
      if (!await activate(current)) throw new Error("The selected workspace could not load. Choose the model again to retry.");
    }
    syncColdStart();
    await refreshStatus();
  }

  // syncColdStart: the landing while no model serves (no capability document: the proxy answers alone) and no tab
  // is open; the picker is the way on, and the Library, the one tab the cold page serves, is a hash away.
  function syncColdStart() {
    const existing = document.getElementById("cold-start");
    if (capabilityDocument || tabs.some((t) => t.panel.classList.contains("active"))) { if (existing) existing.remove(); return; }
    if (existing) return;
    document.getElementById("panels").appendChild(el("div", { class: "card front-empty", id: "cold-start" }, el("h2", { text: "no model serves" }),
      el("div", { class: "note", text: "choose a model from the model pill; the Library tab registers a local model or declares a hosted provider" }),
      el("div", { class: "starters" }, el("button", { class: "btn", text: "Choose a model", onclick: () => document.getElementById("model-pill").click() }),
        el("button", { class: "btn alt", text: "Open the Library", onclick: () => activate("library") }))));
  }

  function activate(id) {
    const tab = tabs.find((t) => t.id === id);
    if (!tab) return;
    closeNavigation();
    document.querySelector(".shell").classList.toggle("conversation-view", id === "chat");
    activeSection = tabSection(tab);
    if (id !== "chat") setFront(false);
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
    syncColdStart();
    if (location.hash.slice(1) !== id) history.replaceState(null, "", "#" + id);
    return tab.ready || Promise.resolve(true);
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

  // embed: a registered tab mounted into another host with a seed (the inspector opens analysis tabs over one turn).
  function embed(id, host, seed) { const tab = tabs.find((t) => t.id === id); if (!tab) throw new Error("no workspace tab " + id); clear(host); return tab.mount(host, window.overgo, seed); }
  // analysisSurface: the shared inspector head (seeded prompt, labelled fields, run/cancel, output host).
  function analysisSurface(panel, seed, options) {
    const prompt = el("textarea", { class: "text", placeholder: "prompt to analyze…" });
    prompt.value = (seed && seed.prompt) || options.defaultPrompt;
    const run = el("button", { class: "btn", onclick: () => surface.execute() }, options.runLabel);
    const cancel = el("button", { class: "btn alt", hidden: true }, "cancel");
    const out = el("div");
    panel.append(prompt, el("div", { class: "row my-10" }, ...options.fields.flatMap(([label, input]) => [el("span", { class: "note", text: label }), input]), run, cancel),
      ...(options.note ? [el("div", { class: "note", text: options.note })] : []), out);
    const runAction = runner(run, cancel, {
      onError: (err) => out.replaceChildren(errorBanner(friendlyError(err))),
      onCancel: () => out.replaceChildren(el("div", { class: "note", text: "[cancelled]" })),
    });
    const surface = { prompt, out, execute() { out.replaceChildren(el("div", { class: "note", text: options.busy })); runAction(options.execute); } };
    return surface;
  }
  function safeMount(tab) {
    const attempt = {};
    tab.mountAttempt = attempt;
    const failed = (err) => { if (tab.mountAttempt === attempt) renderMountError(tab, err); return false; };
    try {
      const result = tab.mount(tab.panel, window.overgo);
      tab.ready = Promise.resolve(result).then(() => tab.mountAttempt === attempt, failed);
    } catch (err) { tab.ready = Promise.resolve(failed(err)); }
  }
  function renderMountError(tab, err) { tab.panel.replaceChildren(errorBanner(String(err && err.message || err))); }

  // dot: one header status dot (server, swap proxy, device) with its state and its fact as the title.
  function dot(id, state, title) {
    const node = document.getElementById(id);
    if (node) { node.className = "dot " + state; node.title = title; }
  }
  let statusAttempt = 0;
  async function refreshStatus() {
    const attempt = ++statusAttempt;
    const statusPill = document.getElementById("status-pill");
    const modelPill = document.getElementById("model-pill");
    try {
      // The proxy names its child in the header; an empty name is the proxy with no child (the cold start).
      const health = await api.get("/health", { onHeaders: (headers) => { if (attempt !== statusAttempt) return; const via = headers.get("X-Overgo-Swap-Proxy");
        dot("proxy-dot", via == null ? "off" : "ok", via ? "swap proxy serving " + via : via == null ? "no swap proxy: served directly" : "swap proxy running; no model serves"); } });
      if (attempt !== statusAttempt) return;
      statusPill.textContent = "online";
      document.getElementById("connection-alert").hidden = true;
      statusPill.className = "pill ok";
      dot("server-dot", "ok", "server online");
      dot("device-dot", health.device ? "ok" : "off", health.device ? "device peak " + window.overgo.fmt.bytes(health.device.peak_bytes) + " · current " + window.overgo.fmt.bytes(health.device.current_bytes) : "no device");
      // The proxy with no child names no model; the pill says so rather than keeping the last name.
      modelPill.textContent = (health && health.model) || "no model serves";
      modelPill.title = modelPill.textContent + " — choose a model";
      document.getElementById("model-details").textContent = modelPill.textContent;
      servedEntry = null;
      document.getElementById("model-evidence").textContent = "";
      if (health && health.model) {
        // A catalog that fails to list says so under the pill instead of an empty evidence line.
        const catalog = await api.get("/catalog/models").catch((err) => ({ models: [], refusal: "catalog: " + friendlyError(err) }));
        if (attempt !== statusAttempt) return;
        servedEntry = catalog.models.find((item) => (item.location || "").split(/[\\/]/).pop() === health.model || item.model === health.model) || null;
        document.getElementById("model-evidence").textContent = servedEntry ? evidenceLine(servedEntry) : catalog.refusal || "";
      }
    } catch (err) { if (attempt !== statusAttempt) return; statusPill.textContent = "offline"; statusPill.className = "pill err"; dot("server-dot", "err", "server offline"); document.getElementById("connection-alert").hidden = false; }
  }

  // The served model shows on every page as the banner pill; clicking it lists the servable models, and behind
  // the swap proxy choosing one swaps the serving child live while every page keeps working.
  // libraryStarters: the two ways a model enters an empty store, each a control that opens the Library tab
  // on its form (the local registration row, the hosted provider form) once the tab has mounted.
  function libraryStarters() {
    const open = (selector) => { location.hash = "#library"; const focus = () => { const field = document.querySelector(selector); if (field) field.focus(); else setTimeout(focus, focusRetryMS); }; focus(); };
    return [el("button", { class: "btn alt", text: "register a local model", onclick: () => open("input[placeholder='model GGUF or directory on disk']") }),
      el("button", { class: "btn alt", text: "declare a hosted provider", onclick: () => open("input[aria-label='provider name']") })];
  }
  function wireModelPicker() {
    const modelPill = document.getElementById("model-pill");
    if (!modelPill) return;
    let panel = null; modelPill.classList.add("clickable");
    modelPill.title = "click to switch the served model";
    // close: the picker leaves after a served swap and on Escape; a click on the pill toggles it.
    const close = () => { if (panel) { panel.close(); panel.remove(); panel = null; modelPill.focus(); } };

    // swapModel routes one health probe through the swap proxy with the swap query parameter; the proxy swaps
    // the child to answer it, then the capability document is re-read and the active surface re-mounted.
    async function swapModel(item, name, button) {
      if (modelSwitchPending) return;
      modelSwitchPending = true;
      setModelSwitchBlocked(true);
      const currentPanel = panel;
      const before = capabilityDocument;
      const started = Date.now();
      const chip = { id: "swap-" + name, task: "model switch " + name, state: "running", progress: {} };
      window.overgo.localOperation(chip);
      button.disabled = true;
      for (const choice of currentPanel.querySelectorAll('[data-serve]')) choice.disabled = true;
      const timer = setInterval(() => { button.textContent = "loading… " + Math.round((Date.now() - started) / 1000) + "s"; }, loadingTickMS);
      try {
        await api.get("/health?swap=" + encodeURIComponent(name));
        invalidateModel();
        const manifest = await api.get("/workspace/manifest");
        if (manifest.model && manifest.model.recipe === item.recipe) {
          try { localStorage.setItem(MODEL_STORAGE, name); } catch (_) { /* storage unavailable */ }
          // Reopening history can request its original model; other switches start a fresh chain.
          if (!selectedConversation || selectedConversation.model !== manifest.model.model) rememberConversation(null);
          await remountActive(manifest);
          setModelSwitchBlocked(false);
          // The served swap is the operation chip's receipt; the picker has done its work and leaves.
          const elapsed = Math.round((Date.now() - started) / 1000);
          window.overgo.localOperation(Object.assign(chip, { state: "completed", detail: "served " + modelPill.textContent + " after " + elapsed + "s" +
            (capabilityDocument ? " · context " + capabilityDocument.context_length + " · " + Object.keys(capabilityDocument.modalities || {}).filter((kind) => capabilityDocument.modalities[kind]).join(", ") : "") }));
          if (panel === currentPanel) close();
          return;
        }
        if (before && manifest.model && before.recipe === manifest.model.recipe) setModelSwitchBlocked(false);
        window.overgo.localOperation(Object.assign(chip, { state: "failed", failure: "the requested model was not served" }));
        const command = 'overgo_gui.bat "' + (item.location || name) + '"';
        const copy = el("button", { class: "btn alt", text: "copy launch" });
        copy.addEventListener("click", () => navigator.clipboard.writeText(command));
        if (panel !== currentPanel) return;
        panel.replaceChildren(
          el("div", { class: "note", text: "this server runs without the swap proxy; relaunch it on the model instead" }),
          el("div", { class: "row" }, el("span", { class: "mono", text: command }), copy));
      } catch (err) {
        // A refused swap may leave the original model usable. Confirm its
        // recipe before releasing Send; an unknown model stays blocked.
        try {
          const current = await api.get("/workspace/manifest");
          if (before && current.model && before.recipe === current.model.recipe) {
            await remountActive(current);
            setModelSwitchBlocked(false);
          }
        } catch (_) { /* choose a model again to resolve its unknown state */ }
        window.overgo.localOperation(Object.assign(chip, { state: "failed", failure: friendlyError(err) }));
        if (panel === currentPanel) panel.replaceChildren(errorBanner(friendlyError(err)), el("button", { class: "btn alt", text: "Choose model again", onclick: () => { close(); modelPill.click(); } }), el("button", { class: "btn alt", text: "Close", onclick: close }));
      } finally {
        modelSwitchPending = false;
        clearInterval(timer); button.textContent = "serve";
        if (panel) for (const choice of panel.querySelectorAll('[data-serve]')) choice.disabled = false;
      }
    }

    modelPill.addEventListener("click", async () => {
      if (panel) { close(); return; }
      panel = el("dialog", { class: "card picker-card", "aria-label": "Choose a model" });
      const currentPanel = panel;
      modelPill.parentElement.appendChild(panel);
      panel.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
      panel.showModal();
      panel.textContent = "loading servable models…";
      try {
        const catalog = await api.get("/catalog/models");
        if (panel !== currentPanel) return;
        // Every activated entry with bytes on disk is listed: a servable one with its evidence, declared
        // task capabilities and a serve control; a stale activation with the loader's reason and no control.
        const entries = (catalog.models || []).filter((item) => item.present && item.recipe);
        // An empty store names the two ways in; each opens the Library tab's form.
        if (!entries.length) {
          const starters = libraryStarters();
          for (const starter of starters) starter.addEventListener("click", close);
          panel.replaceChildren(el("div", { class: "note", text: "Add a local model or connect a hosted provider." }), el("div", { class: "row" }, ...starters));
          return;
        }
        let remembered = "";
        try { remembered = localStorage.getItem(MODEL_STORAGE) || ""; } catch (_) { /* storage unavailable */ }
        panel.replaceChildren(el("div", { class: "note", text: "switch the served model; the load can take a minute" + (remembered && remembered !== modelPill.textContent ? " · last time you served " + remembered : "") }),
          ...entries.map((item) => {
            const name = item.location ? item.location.split(/[\\/]/).pop() : item.model;
            // Keyboard path: a focused row serves on Enter; the arrows move between rows.
            const row = el("div", { class: "row", tabindex: "0", onkeydown: (event) => {
              if (event.key === "Enter" && event.target === row) { const serve = [...row.querySelectorAll("button")].find((button) => button.textContent === "serve"); if (serve) serve.click(); }
              else if (event.key === "ArrowDown" || event.key === "ArrowUp") { const rows = [...currentPanel.querySelectorAll(".row[tabindex]")], next = rows[rows.indexOf(row) + (event.key === "ArrowDown" ? 1 : -1)]; if (next) { event.preventDefault(); next.focus(); } }
            } }, el("span", { class: "mono", text: name }));
            const facts = el("details", { class: "picker-facts" }, el("summary", { text: "Details" })); row.appendChild(facts);
            if ((item.location || "").startsWith("remote://")) facts.appendChild(el("span", { class: "tag", title: "served at a hosted provider through the relay", text: "remote" }));
            const evidence = evidenceLine(item);
            if (evidence) facts.appendChild(el("span", { class: "note", text: evidence }));
            for (const capability of item.capabilities || []) facts.appendChild(el("span", { class: "tag", text: capability.task + (capability.tier ? " · " + capability.tier : "") }));
            if (item.stale) {
              facts.open = true;
              facts.appendChild(el("span", { class: "tag tag-danger", title: item.stale, text: "unservable: " + item.stale }));
              // A keyless hosted model takes its key here; the proxy and the served child hold it in memory only, and the picker relists.
              if (item.key_environment) {
                const key = el("input", { class: "keyfield", type: "password", placeholder: item.key_environment, "aria-label": "provider key" });
                facts.append(key, el("button", { class: "btn alt", text: "use key", onclick: async () => {
                  try { await api.post("/providers/key", { location: item.location, key: key.value }); if (panel === currentPanel) { close(); modelPill.click(); } } catch (err) { if (panel === currentPanel) panel.textContent = friendlyError(err); }
                } }));
              }
              return row;
            }
            row.appendChild(el("button", { class: "btn alt", text: "serve", "data-serve": "", disabled: modelSwitchPending, onclick: (event) => swapModel(item, name, event.currentTarget) })); return row;
          }));
        const first = panel.querySelector(".row"); if (first) first.focus();
      } catch (err) { if (panel === currentPanel) panel.textContent = friendlyError(err); }
      finally {
        if (panel === currentPanel) panel.prepend(el("div", { class: "dialog-heading" }, el("h2", { text: "Choose a model" }), el("button", { class: "btn alt", text: "Close", onclick: close })));
      }
    });
    // Alt+M opens the picker from anywhere on the page; the first row takes focus.
    modelPill.setAttribute("aria-keyshortcuts", "Alt+M");
    document.addEventListener("keydown", (event) => { if (event.altKey && !event.ctrlKey && event.key.toLowerCase() === "m") { event.preventDefault(); modelPill.click(); } });
  }
  wireModelPicker();

  function tabSupported(tab) { return tab.enabled; }

  function applyCapabilities() {
    for (const tab of tabs) {
      const ok = tabSupported(tab);
      // A capability this model does not serve HIDES its tab: the nav shows what works here, and the
      // manifest still carries every refusal for API clients that ask.
      tab.button.hidden = !ok;
      tab.button.disabled = !ok;
      tab.button.title = ok ? "" : tab.refusal;
    }
    const active = tabs.find((t) => t.button.classList.contains("active"));
    if (active && !tabSupported(active)) { const firstOk = tabs.find(tabSupported); if (firstOk) activate(firstOk.id); }
  }

  function bindWorkspaceManifest(manifest) {
    if (!manifest || !Array.isArray(manifest.sections) || !Array.isArray(manifest.tabs)) throw new Error("Workspace manifest is invalid");
    const implementations = new Map(tabs.map((tab) => [tab.id, tab]));
    for (const id of implementations.keys()) if (!manifest.tabs.some((tab) => tab.id === id)) throw new Error("Undeclared workspace tab " + id);
    const ordered = [];
    for (const declaration of manifest.tabs) {
      const implementation = implementations.get(declaration.id);
      if (!implementation) { errors.push("Workspace tab " + declaration.id + " did not register from its module"); continue; }
      ordered.push(Object.assign(implementation, declaration));
    }
    tabs.splice(0, tabs.length, ...ordered);
  }

  // ---- one loader: the manifest names each tab's module (default: the tab id); the libraries load in
  // order, then every distinct module, then the shell wires. Same-origin scripts, so the strict CSP holds. ----
  const libraries = ["/viz.js", "/md.js", "/composer.js", "/workflow.js", "/operations_shell.js", "/schema_form.js"];
  const loadedScripts = new Set();
  function loadScript(src) {
    if (loadedScripts.has(src)) return Promise.resolve(src);
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.async = false; // insertion order is execution order
      script.onload = () => { loadedScripts.add(src); resolve(src); };
      script.onerror = () => reject(new Error("failed to load " + src));
      document.head.appendChild(script);
    });
  }
  async function loadWorkspaceModules(manifest) {
    for (const src of libraries) await loadScript(src);
    const modules = [...new Set(manifest.tabs.map((declaration) => declaration.module || declaration.id))];
    await Promise.all(modules.map((module) => loadScript("/mod/" + module + ".js")));
  }

  // offlineCard: the landing state while the server does not answer; it
  // probes /health until it does, then initialises the shell in place.
  let offlineTimer = null;
  function offlineCard(panels, message) {
    clear(panels);
    const dot = el("span", { class: "dot err" });
    const text = el("span", { text: message });
    const retry = el("button", { class: "btn alt", text: "probe again" });
    const card = el("div", { class: "card" }, el("div", { class: "probe" }, dot, text),
      el("p", { class: "note mt-18", text: "Start the server (overgo_gui.bat, or cmd/server with a model) and this page enters on its own." }),
      el("div", { class: "actions" }, retry));
    panels.appendChild(el("div", { class: "center tall" }, card));
    async function probe() {
      dot.className = "dot scan";
      try {
        await api.get("/health");
        dot.className = "dot ok";
        text.textContent = "server online, entering…";
        clearInterval(offlineTimer);
        offlineTimer = null;
        initShell();
      } catch (err) { dot.className = "dot err"; text.textContent = "no server at " + location.origin + " (" + friendlyError(err) + ")"; }
    }
    retry.addEventListener("click", probe);
    if (offlineTimer == null) offlineTimer = setInterval(probe, offlineProbeMS);
  }

  // The served model's capability document rides the manifest; every client capability decision reads capabilities().
  let capabilityDocument = null;
  function capabilities() { return capabilityDocument; }
  window.overgo.capabilities = capabilities;

  async function initShell() {
    const sectionBar = document.getElementById("sections");
    const panels = document.getElementById("panels");
    try {
      workspaceManifest = await api.get("/workspace/manifest");
      capabilityDocument = workspaceManifest.model || null;
    } catch (err) { offlineCard(panels, "no server at " + location.origin + " (" + friendlyError(err) + ")"); return; }
    clear(panels);
    clear(sectionBar);
    sectionButtons.length = 0;
    try {
      await loadWorkspaceModules(workspaceManifest);
      bindWorkspaceManifest(workspaceManifest);
    } catch (err) { panels.appendChild(errorBanner(friendlyError(err))); return; }
    for (const section of sectionsPresent()) {
      const button = el("button", { class: "section", onclick: () => selectSection(section.id) }, section.label);
      const group = el("div", { class: "nav-group" }, button);
      sectionButtons.push({ id: section.id, button: button });
      for (const tab of tabs) {
        if (tabSection(tab) !== section.id) continue;
        tab.button = el("button", { class: "tab", onclick: () => activate(tab.id) }, tab.label);
        tab.panel = el("div", { class: "panel", id: "panel-" + tab.id });
        group.appendChild(tab.button);
        panels.appendChild(tab.panel);
      }
      sectionBar.appendChild(group);
    }
    // The key controls, hash routing and the health re-probe are wired once; the shell may initialise
    // again after an offline card and must not stack a second listener.
    if (!shellWired) {
      shellWired = true;
      document.getElementById("workbench-toggle").addEventListener("click", () =>
        setFront(!document.querySelector(".shell").classList.contains("front")));
      const keyInput = document.getElementById("api-key");
      const keyRemember = document.getElementById("api-key-remember");
      keyInput.value = getKey();
      keyRemember.checked = keyWasPersisted;
      keyRemember.addEventListener("change", () => { setKey(keyInput.value.trim(), keyRemember.checked); });
      keyInput.addEventListener("change", () => {
        setKey(keyInput.value.trim(), keyRemember.checked);
        invalidateModel(); // the cached model was fetched under the old key
        remountActive().catch(err => { document.getElementById("connection-alert").hidden = false; document.getElementById("conversation-settings").appendChild(errorBanner(friendlyError(err))); });
      });
      window.addEventListener("hashchange", () => { const id = location.hash.slice(1); if (id && tabs.some((t) => t.id === id)) activate(id); });
      // Re-probe health so a server that drops (or comes back) is reflected in the
      // status pill instead of showing a stale "online" until the next key change.
      setInterval(refreshStatus, statusRefreshMS);
    }
    const start = location.hash.slice(1);
    applyCapabilities();
    const named = tabs.some((t) => t.id === start);
    if (named || capabilityDocument) activate(named ? start : (tabs[0] && tabs[0].id));
    syncColdStart();
    refreshStatus();
    refreshConversations();
  }
  let shellWired = false;

  // boot.js is deferred: the document is parsed, and the loader brings in
  // every library and module itself, so the shell starts at once.
  initShell();
})();
