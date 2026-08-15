/* Vocabulary tab: renders GET /analyze/vocab — a paged, searchable listing of
   raw vocabulary entries (id, text, GGUF type flag, score). Pure enumeration. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "vocab",
    label: "Vocabulary",
    section: "workbench",
    requires: "vocabulary",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      const limit = 128;
      let offset = 0;
      let query = "";
      let timer = null;

      clear(panel);
      const search = el("input", {
        class: "text", type: "search", placeholder: "search token text (substring, case-insensitive)…",
        style: "max-width:420px",
      });
      const status = el("span", { class: "note" });
      const prev = el("button", { class: "btn alt", onclick: () => { offset = Math.max(0, offset - limit); load(); } }, "‹ prev");
      const next = el("button", { class: "btn alt", onclick: () => { offset += limit; load(); } }, "next ›");
      const controls = el("div", { class: "row", style: "margin-bottom:12px" }, search, prev, next, status);
      const host = el("div");
      panel.append(controls, host);

      search.addEventListener("input", () => {
        clearTimeout(timer);
        timer = setTimeout(() => { query = search.value.trim(); offset = 0; load(); }, 180);
      });

      async function load() {
        host.replaceChildren(el("div", { class: "note", text: "loading…" }));
        const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
        if (query) params.set("query", query);
        let data;
        try {
          data = await overgo.api.get("/analyze/vocab?" + params.toString());
        } catch (err) {
          host.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
          return;
        }
        // Clamp a past-the-end page back to the last populated window.
        if (data.tokens.length === 0 && offset > 0 && data.matched > 0) {
          offset = Math.max(0, Math.floor((data.matched - 1) / limit) * limit);
          return load();
        }
        const first = data.matched === 0 ? 0 : offset + 1;
        const last = offset + data.tokens.length;
        status.textContent = query
          ? first + "–" + last + " of " + fmt.grouped(data.matched) + " matched (" + fmt.grouped(data.size) + " total)"
          : first + "–" + last + " of " + fmt.grouped(data.size);
        prev.disabled = offset === 0;
        next.disabled = last >= data.matched;

        const table = el("table", { class: "grid" });
        table.appendChild(el("tr", {},
          el("th", { text: "id" }), el("th", { text: "token" }), el("th", { text: "type" }), el("th", { text: "score" })));
        for (const token of data.tokens) {
          table.appendChild(el("tr", {},
            el("td", { class: "mono", text: String(token.id) }),
            el("td", { class: "mono", text: token.text === "" ? "∅" : token.text }),
            el("td", {}, el("span", { class: "tag " + token.type, text: token.type })),
            el("td", { class: "mono", text: token.score ? token.score.toFixed(4) : "0" })));
        }
        host.replaceChildren(table);
      }

      await load();
    },
  });
})();
