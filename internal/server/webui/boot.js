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
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data.push(line.slice(6));
        }
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

  function errorBanner(message) { return el("div", { class: "err-banner", text: message }); }

  // friendlyError: a 401 becomes the same actionable hint on every tab.
  function friendlyError(err) { return err && err.status === 401 ? "API key required — enter it in the top bar." : String((err && err.message) || err); }

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
  function displayToken(text) {
    if (text === "" || text == null) return "∅";
    if (/^\s+$/.test(text)) return "␠".repeat(text.length);
    return text.replace(/\n/g, "⏎");
  }

  // runner: one exclusive, cancelable async action bound to a run and a cancel
  // button; an abort goes to onCancel, any other failure to onError.
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
      catch (err) { if (!err || err.name !== "AbortError") throw err; } finally {
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
  function artifactLink(id, label, gallery) {
    return el("a", {
      class: "mono", href: (gallery ? "/artifacts?id=" : "/artifacts/content?id=") + encodeURIComponent(id),
      target: "_blank", rel: "noopener", text: label || shortID(id),
    });
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
  function setFront(on) { document.querySelector(".shell").classList.toggle("front", on); }
  let servedEntry = null;
  function servedModel() { return servedEntry; }

  // ---- conversations: the server lists stored-response chains; the rail opens, renames or archives one ----
  let selectedConversation = null;
  function conversation() { return selectedConversation; }
  function openConversation(item) {
    selectedConversation = item;
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
      const rename = el("button", { class: "link-button", text: "rename", "aria-label": "rename conversation",
        onclick: () => { const title = window.prompt("conversation title", item.title); if (title != null) label({ title }); } });
      const archive = el("button", { class: "link-button", text: "archive", "aria-label": "archive conversation", onclick: () => label({ archived: true }) });
      host.appendChild(el("div", { class: "conversation" }, open, el("div", { class: "row" }, rename, archive)));
    }
  }

  window.overgo = {
    api, el, clear, errorBanner, friendlyError, registerTab, artifactLink, headerRow, tableRow, table, evidenceLine, servedModel,
    conversation, openConversation, refreshConversations, sseEvents, errors, embed, analysisSurface, reporter,
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
  function syncSectionUI() {
    for (const sb of sectionButtons) sb.button.classList.toggle("active", sb.id === activeSection);
  }

  // remountActive: the active tab reloads under a new key, a newly served model or a
  // changed store, from the capability document re-read for it.
  async function remountActive() {
    try { workspaceManifest = await api.get("/workspace/manifest"); capabilityDocument = workspaceManifest.model || null; } catch (_) { /* the shell stays on what it has */ }
    for (const tab of tabs) {
      if (tab.onDeactivate) tab.onDeactivate();
      tab.mounted = false;
    }
    const current = location.hash.slice(1) || (tabs[0] && tabs[0].id);
    if (current) activate(current);
    refreshStatus();
  }

  function activate(id) {
    const tab = tabs.find((t) => t.id === id);
    if (!tab) return;
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

  // embed: a registered tab mounted into another host with a seed (the inspector opens analysis tabs over one turn).
  function embed(id, host, seed) {
    const tab = tabs.find((t) => t.id === id);
    if (!tab) throw new Error("no workspace tab " + id);
    clear(host);
    return tab.mount(host, window.overgo, seed);
  }
  // analysisSurface: the shared inspector head (seeded prompt, labelled fields, run/cancel, output host).
  function analysisSurface(panel, seed, options) {
    const prompt = el("textarea", { class: "text", placeholder: "prompt to analyze…" });
    prompt.value = (seed && seed.prompt) || options.defaultPrompt;
    const run = el("button", { class: "btn", onclick: () => surface.execute() }, options.runLabel);
    const cancel = el("button", { class: "btn alt", style: "display:none" }, "cancel");
    const out = el("div");
    panel.append(prompt, el("div", { class: "row", style: "margin:10px 0" }, ...options.fields.flatMap(([label, input]) => [el("span", { class: "note", text: label }), input]), run, cancel),
      ...(options.note ? [el("div", { class: "note", text: options.note })] : []), out);
    const runAction = runner(run, cancel, {
      onError: (err) => out.replaceChildren(errorBanner(friendlyError(err))),
      onCancel: () => out.replaceChildren(el("div", { class: "note", text: "[cancelled]" })),
    });
    const surface = { prompt, out, execute() { out.replaceChildren(el("div", { class: "note", text: options.busy })); runAction(options.execute); } };
    return surface;
  }
  function safeMount(tab) {
    try {
      const result = tab.mount(tab.panel, window.overgo);
      if (result && typeof result.catch === "function") result.catch((err) => renderMountError(tab, err));
    } catch (err) { renderMountError(tab, err); }
  }
  function renderMountError(tab, err) { tab.panel.replaceChildren(errorBanner(String(err && err.message || err))); }

  // dot: one header status dot (server, swap proxy, device) with its state and its fact as the title.
  function dot(id, state, title) {
    const node = document.getElementById(id);
    if (node) { node.className = "dot " + state; node.title = title; }
  }
  async function refreshStatus() {
    const statusPill = document.getElementById("status-pill");
    const modelPill = document.getElementById("model-pill");
    try {
      const health = await api.get("/health", { onHeaders: (headers) => dot("proxy-dot", headers.has("X-Overgo-Swap-Proxy") ? "ok" : "off",
        headers.has("X-Overgo-Swap-Proxy") ? "swap proxy serving " + headers.get("X-Overgo-Swap-Proxy") : "no swap proxy: served directly") });
      statusPill.textContent = "online";
      statusPill.className = "pill ok";
      dot("server-dot", "ok", "server online");
      dot("device-dot", health.device ? "ok" : "off", health.device ? "device peak " + window.overgo.fmt.bytes(health.device.peak_bytes) + " · current " + window.overgo.fmt.bytes(health.device.current_bytes) : "no device");
      if (health && health.model) {
        modelPill.textContent = health.model;
        const catalog = await api.get("/catalog/models").catch(() => null);
        servedEntry = ((catalog && catalog.models) || []).find((item) =>
          (item.location || "").split(/[\\/]/).pop() === health.model || item.model === health.model) || null;
        document.getElementById("model-evidence").textContent = servedEntry ? evidenceLine(servedEntry) : "";
      }
    } catch (err) {
      statusPill.textContent = "offline";
      statusPill.className = "pill err";
      dot("server-dot", "err", "server offline");
    }
  }

  // The served model shows on every page as the banner pill; clicking it lists the servable models, and behind
  // the swap proxy choosing one swaps the serving child live while every page keeps working.
  // libraryStarters: the two ways a model enters an empty store, each a control that opens the Library tab
  // on its form (the local registration row, the hosted provider form) once the tab has mounted.
  function libraryStarters() {
    const open = (selector) => { location.hash = "#library"; const focus = () => { const field = document.querySelector(selector); if (field) field.focus(); else setTimeout(focus, 50); }; focus(); };
    return [el("button", { class: "btn alt", text: "register a local model", onclick: () => open("input[placeholder='model GGUF or directory on disk']") }),
      el("button", { class: "btn alt", text: "declare a hosted provider", onclick: () => open("input[aria-label='provider name']") })];
  }
  function wireModelPicker() {
    const modelPill = document.getElementById("model-pill");
    if (!modelPill) return;
    let panel = null;
    modelPill.style.cursor = "pointer";
    modelPill.title = "click to switch the served model";

    // swapModel routes one health probe through the swap proxy with the swap query parameter; the proxy swaps
    // the child to answer it, then the capability document is re-read and the active surface re-mounted.
    async function swapModel(item, name, button) {
      const before = modelPill.textContent;
      const started = Date.now();
      const chip = { id: "swap-" + name, task: "model switch " + name, state: "running", progress: {} };
      window.overgo.localOperation(chip);
      button.disabled = true;
      const timer = setInterval(() => { button.textContent = "loading… " + Math.round((Date.now() - started) / 1000) + "s"; }, 1000);
      try {
        await api.get("/health?swap=" + encodeURIComponent(name));
        invalidateModel();
        await refreshStatus();
        if (modelPill.textContent !== before) {
          try { localStorage.setItem(MODEL_STORAGE, name); } catch (_) { /* storage unavailable */ }
          workspaceManifest = await api.get("/workspace/manifest");
          capabilityDocument = workspaceManifest.model || null;
          remountActive();
          window.overgo.localOperation(Object.assign(chip, { state: "completed", detail: "served after " + Math.round((Date.now() - started) / 1000) + "s" }));
          const elapsed = Math.round((Date.now() - started) / 1000);
          panel.replaceChildren(el("div", { class: "note", text: "now serving " + modelPill.textContent + " after " + elapsed + "s · " +
            (capabilityDocument ? "context " + capabilityDocument.context_length + " · " + Object.keys(capabilityDocument.modalities || {}).filter((kind) => capabilityDocument.modalities[kind]).join(", ") : "") }));
          return;
        }
        window.overgo.localOperation(Object.assign(chip, { state: "failed", failure: "no swap proxy" }));
        const command = 'overgo_gui.bat "' + (item.location || name) + '"';
        const copy = el("button", { class: "btn alt", text: "copy launch" });
        copy.addEventListener("click", () => navigator.clipboard.writeText(command));
        panel.replaceChildren(
          el("div", { class: "note", text: "this server runs without the swap proxy; relaunch it on the model instead" }),
          el("div", { class: "row" }, el("span", { class: "mono", text: command }), copy));
      } catch (err) {
        window.overgo.localOperation(Object.assign(chip, { state: "failed", failure: friendlyError(err) }));
        panel.replaceChildren(errorBanner(friendlyError(err)));
      } finally {
        clearInterval(timer);
        button.disabled = false;
        button.textContent = "serve";
      }
    }

    modelPill.addEventListener("click", async () => {
      if (panel) { panel.remove(); panel = null; return; }
      panel = el("div", { class: "card", style: "position:absolute;right:12px;top:44px;z-index:40;max-width:560px" });
      modelPill.parentElement.appendChild(panel);
      panel.textContent = "loading servable models…";
      try {
        const catalog = await api.get("/catalog/models");
        // Every activated entry with bytes on disk is listed: a servable one with its evidence, declared
        // task capabilities and a serve control; a stale activation with the loader's reason and no control.
        const entries = (catalog.models || []).filter((item) => item.present && item.recipe);
        // An empty store names the two ways in; each opens the Library tab's form.
        if (!entries.length) { panel.replaceChildren(el("div", { class: "note", text: "no activated models in the store — register a local model, or declare a hosted provider and enter its key" }), el("div", { class: "row" }, ...libraryStarters())); return; }
        let remembered = "";
        try { remembered = localStorage.getItem(MODEL_STORAGE) || ""; } catch (_) { /* storage unavailable */ }
        panel.replaceChildren(el("div", { class: "note", text: "switch the served model; the load can take a minute" + (remembered && remembered !== modelPill.textContent ? " · last time you served " + remembered : "") }),
          ...entries.map((item) => {
            const name = item.location ? item.location.split(/[\\/]/).pop() : item.model;
            // Keyboard path: a focused row serves on Enter; the arrows move between rows.
            const row = el("div", { class: "row", tabindex: "0", onkeydown: (event) => {
              if (event.key === "Enter") { const serve = [...row.querySelectorAll("button")].find((button) => button.textContent === "serve"); if (serve) serve.click(); }
              else if (event.key === "ArrowDown" || event.key === "ArrowUp") { const rows = [...panel.querySelectorAll(".row")], next = rows[rows.indexOf(row) + (event.key === "ArrowDown" ? 1 : -1)]; if (next) { event.preventDefault(); next.focus(); } }
            } }, el("span", { class: "mono", text: name }));
            if ((item.location || "").startsWith("remote://")) row.appendChild(el("span", { class: "tag", title: "served at a hosted provider through the relay", text: "remote" }));
            const facts = evidenceLine(item);
            if (facts) row.appendChild(el("span", { class: "note", text: facts }));
            for (const capability of item.capabilities || []) row.appendChild(el("span", { class: "tag", text: capability.task + (capability.tier ? " · " + capability.tier : "") }));
            if (item.stale) {
              row.appendChild(el("span", { class: "tag tag-danger", title: item.stale, text: "unservable: " + item.stale }));
              // A keyless hosted model takes its key here; the proxy and the served child hold it in memory only, and the picker relists.
              if (item.key_environment) {
                const key = el("input", { class: "keyfield", type: "password", placeholder: item.key_environment, "aria-label": "provider key" });
                row.append(key, el("button", { class: "btn alt", text: "use key", onclick: async () => {
                  try { await api.post("/providers/key", { location: item.location, key: key.value }); modelPill.click(); modelPill.click(); } catch (err) { panel.textContent = friendlyError(err); }
                } }));
              }
              return row;
            }
            row.appendChild(el("button", { class: "btn alt", text: "serve", onclick: (event) => swapModel(item, name, event.currentTarget) })); return row;
          }));
        const first = panel.querySelector(".row"); if (first) first.focus();
      } catch (err) { panel.textContent = friendlyError(err); }
    });
    // Alt+M opens the picker from anywhere on the page; the first row takes focus.
    modelPill.setAttribute("aria-keyshortcuts", "Alt+M");
    document.addEventListener("keydown", (event) => { if (event.altKey && !event.ctrlKey && event.key.toLowerCase() === "m") { event.preventDefault(); modelPill.click(); } });
  }
  wireModelPicker();

  // ---- capability gating ----
  // Refusal is server-owned and travels with the same manifest as navigation.
  let authNoticeEl = null;

  // When /analyze/model answers 401 the whole analysis surface is locked behind the key: one banner and
  // a highlighted field instead of every tab failing on its own with a raw bearer-token error.
  function showAuthNotice(show) {
    if (authNoticeEl) authNoticeEl.style.display = show ? "" : "none";
    const key = document.getElementById("api-key"); if (key) key.classList.toggle("needs-key", show);
  }

  function tabSupported(tab) { return tab.enabled; }

  function applyCapabilities() {
    for (const tab of tabs) {
      const ok = tabSupported(tab);
      // A capability this model does not serve HIDES its tab: the nav shows what works here, and the
      // manifest still carries every refusal for API clients that ask.
      tab.button.style.display = ok ? "" : "none";
      tab.button.disabled = !ok;
      tab.button.title = ok ? "" : tab.refusal;
    }
    const active = tabs.find((t) => t.button.classList.contains("active"));
    if (active && !tabSupported(active)) {
      const firstOk = tabs.find(tabSupported);
      if (firstOk) activate(firstOk.id);
    }
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
      el("p", { class: "note", style: "margin-top:18px", text: "Start the server (overgo_gui.bat, or cmd/server with a model) and this page enters on its own." }),
      el("div", { class: "actions" }, retry));
    panels.appendChild(el("div", { class: "center", style: "min-height:60vh" }, card));
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
    if (offlineTimer == null) offlineTimer = setInterval(probe, 4000);
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
    if (!authNoticeEl) {
      authNoticeEl = el("div", { class: "auth-banner", style: "display:none" },
        "This server requires an API key — enter it in the field at the top right to load analysis and chat.");
      document.querySelector(".wrap").insertBefore(authNoticeEl, panels);
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
      keyRemember.addEventListener("change", () => {
        setKey(keyInput.value.trim(), keyRemember.checked);
      });
      keyInput.addEventListener("change", () => {
        setKey(keyInput.value.trim(), keyRemember.checked);
        invalidateModel(); // the cached model was fetched under the old key
        remountActive();
      });
      window.addEventListener("hashchange", () => {
        const id = location.hash.slice(1);
        if (id && tabs.some((t) => t.id === id)) activate(id);
      });
      // Re-probe health so a server that drops (or comes back) is reflected in the
      // status pill instead of showing a stale "online" until the next key change.
      setInterval(refreshStatus, 10000);
    }
    const start = location.hash.slice(1);
    activate(tabs.some((t) => t.id === start) ? start : (tabs[0] && tabs[0].id));
    refreshStatus();
    applyCapabilities();
    refreshConversations();
  }
  let shellWired = false;

  // boot.js is deferred: the document is parsed, and the loader brings in
  // every library and module itself, so the shell starts at once.
  initShell();
})();
