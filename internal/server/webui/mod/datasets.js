/* Datasets · active RepoDB catalog. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "datasets",
    async mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);
      panel.appendChild(el("div", { class: "note", text: "loading /datasets…" }));

      let data;
      try {
        data = await overgo.api.get("/datasets");
      } catch (err) {
        clear(panel);
        const message = err.status === 501
          ? "Dataset browsing is not configured on this server."
          : overgo.friendlyError(err);
        panel.appendChild(overgo.errorBanner(message));
        return;
      }

      clear(panel);
      panel.appendChild(el("div", { class: "section-title", text: "Dataset registry" }));
      panel.appendChild(el("div", { class: "lens-summary" },
        el("span", { class: "note", text: data.count + " datasets" }),
        el("span", { class: "note mono", text: data.catalog })));

      const search = el("input", {
        class: "text", type: "search", placeholder: "filter by name / source / modality / format…",
        style: "max-width:440px;margin:8px 0",
      });
      const host = el("div");
      const preview = el("div");
      panel.append(search, host, preview);

      async function showPreview(name) {
        try {
          const result = await overgo.api.post("/datasets/preview", { name, position: 0, limit: 1 });
          preview.replaceChildren(
            el("div", { class: "section-title", text: name }),
            ...result.examples.map((example) => el("div", { class: "dataset-example" },
              el("div", { class: "mono", text: example.id }),
              ...example.values.map((value) => el("div", {},
                el("span", { class: "tag", text: value.role + " / " + value.modality }),
                value.text ? el("pre", { class: "preview-text", text: value.text }) :
                  el("span", { class: "note", text: value.encoding + " / " + value.bytes + " bytes" }))))));
        } catch (err) {
          preview.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
        }
      }

      function render() {
        const query = search.value.trim().toLowerCase();
        const rows = (data.datasets || []).filter((d) =>
          !query || [d.name, d.source, d.modality, ...(d.formats || [])].some((v) => (v || "").toLowerCase().includes(query)));
        const table = el("table", { class: "grid" });
        table.appendChild(el("tr", {},
          el("th", { text: "name" }), el("th", { text: "source" }), el("th", { text: "modality" }),
          el("th", { text: "format" }), el("th", { text: "files" }), el("th", { text: "status" })));
        for (const d of rows) {
          table.appendChild(el("tr", {},
            el("td", {}, el("button", { class: "link-button mono", text: d.name, onclick: () => showPreview(d.name) })),
            el("td", {}, el("span", { class: "tag", text: d.source || "—" })),
            el("td", { text: d.modality || "—" }),
            el("td", { text: (d.formats || []).join(", ") || "—" }),
            el("td", { class: "mono", text: String(d.files || 0) }),
            el("td", { class: d.available ? "" : "dim", text: d.available ? "available" : "missing" })));
        }
        host.replaceChildren(el("div", { class: "note", text: rows.length + " shown" }), table);
      }
      search.addEventListener("input", render);
      render();
    },
  });
})();
