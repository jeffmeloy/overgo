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
    async get(path) {
      return readJSON(await fetch(path, { headers: authHeaders() }));
    },
    async post(path, body) {
      return readJSON(await fetch(path, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body),
      }));
    },
  };

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

  const tabs = [];
  function registerTab(tab) { tabs.push(tab); }

  window.overgo = {
    api, el, clear, errorBanner, registerTab,
    getKey, setKey,
    fmt: { grouped, bytes, compact },
  };

  // ---- shell wiring (runs after all deferred module scripts registered) ----
  function activate(id) {
    for (const tab of tabs) {
      const on = tab.id === id;
      tab.button.classList.toggle("active", on);
      tab.panel.classList.toggle("active", on);
      if (on && !tab.mounted) { tab.mounted = true; safeMount(tab); }
    }
    if (location.hash.slice(1) !== id) history.replaceState(null, "", "#" + id);
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

  function initShell() {
    const tabBar = document.getElementById("tabs");
    const panels = document.getElementById("panels");
    for (const tab of tabs) {
      tab.button = el("button", { class: "tab", onclick: () => activate(tab.id) }, tab.label);
      tab.panel = el("div", { class: "panel", id: "panel-" + tab.id });
      tabBar.appendChild(tab.button);
      panels.appendChild(tab.panel);
    }
    const keyInput = document.getElementById("api-key");
    keyInput.value = getKey();
    keyInput.addEventListener("change", () => {
      setKey(keyInput.value.trim());
      // Re-mount the active tab so its data reloads under the new key.
      for (const tab of tabs) tab.mounted = false;
      const current = location.hash.slice(1) || (tabs[0] && tabs[0].id);
      if (current) activate(current);
      refreshStatus();
    });
    window.addEventListener("hashchange", () => {
      const id = location.hash.slice(1);
      if (id && tabs.some((t) => t.id === id)) activate(id);
    });
    const start = location.hash.slice(1);
    activate(tabs.some((t) => t.id === start) ? start : (tabs[0] && tabs[0].id));
    refreshStatus();
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
