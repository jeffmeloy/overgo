/* Datasets · browser. Read-only dataset registry from the server's /datasets
   endpoint (from datasets/manifest.json): name, family, modality, language,
   provenance, with a client-side filter. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "datasets",
    label: "Datasets",
    section: "datasets",
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
          ? "Dataset browsing is not configured on this server (no data root)."
          : overgo.friendlyError(err);
        panel.appendChild(overgo.errorBanner(message));
        return;
      }

      clear(panel);
      panel.appendChild(el("div", { class: "section-title", text: "Dataset registry" }));
      panel.appendChild(el("div", { class: "lens-summary" },
        el("span", { class: "note", text: data.count + " datasets" }),
        el("span", { class: "note mono", text: data.root })));

      const search = el("input", {
        class: "text", type: "search", placeholder: "filter by name / family / modality / language…",
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
          !query || [d.name, d.family, d.modality, d.language].some((v) => (v || "").toLowerCase().includes(query)));
        const table = el("table", { class: "grid" });
        table.appendChild(el("tr", {},
          el("th", { text: "name" }), el("th", { text: "family" }), el("th", { text: "modality" }),
          el("th", { text: "language" }), el("th", { text: "provenance" })));
        for (const d of rows) {
          table.appendChild(el("tr", {},
            el("td", {}, el("button", { class: "link-button mono", text: d.name, onclick: () => showPreview(d.name) })),
            el("td", {}, el("span", { class: "tag", text: d.family || "—" })),
            el("td", { text: d.modality || "—" }),
            el("td", { text: d.language || "—" }),
            el("td", { class: "dim", text: d.provenance || "" })));
        }
        host.replaceChildren(el("div", { class: "note", text: rows.length + " shown" }), table);
      }
      search.addEventListener("input", render);
      render();
    },
  });
})();
