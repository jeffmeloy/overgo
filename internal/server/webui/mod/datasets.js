/* Datasets · active OvergoDB catalog. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "datasets",
    async mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);
      panel.appendChild(el("div", { class: "note", text: "Loading datasets…" }));

      let data;
      try {
        data = await overgo.api.get("/datasets");
      } catch (err) {
        clear(panel);
        // A store with no dataset catalog yet is empty, not failing: name how to publish one.
        if (err.status === 404) {
          panel.append(el("div", { class: "section-title", text: "Dataset registry" }),
            el("div", { class: "note", text: "This store has no dataset catalog yet. Register a directory with: go run ./cmd/dataset-catalog -register <name> -root <dir>" }));
          return;
        }
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

      const search = el("input", { "aria-label": "Filter datasets", class: "text mw-440 my-8", type: "search", placeholder: "filter by name / source / modality / format…", });
      const host = el("div");
      const preview = el("div");
      panel.append(search, host, preview);

      // The latest chosen dataset owns the preview: choosing another aborts the earlier read.
      let previewing = null;
      async function showPreview(name) {
        if (previewing) previewing.abort();
        const current = new AbortController();
        previewing = current;
        try {
          const result = await overgo.api.post("/datasets/preview", { name, position: 0, limit: 1 }, { signal: current.signal });
          if (previewing !== current) return;
          preview.replaceChildren(
            el("div", { class: "section-title", text: name }),
            ...result.examples.map((example) => el("div", { class: "dataset-example" },
              el("div", { class: "mono", text: example.id }),
              ...example.values.map((value) => el("div", {},
                el("span", { class: "tag", text: value.role + " / " + value.modality }),
                value.text ? el("pre", { class: "preview-text", text: value.text }) :
                  el("span", { class: "note", text: value.encoding + " / " + value.bytes + " bytes" }))))));
        } catch (err) { if (previewing === current && err.name !== "AbortError") preview.replaceChildren(overgo.failure(err)); }
      }

      function render() {
        const query = search.value.trim().toLowerCase();
        const rows = (data.datasets || []).filter((d) =>
          !query || [d.name, d.source, d.modality, ...(d.formats || [])].some((v) => (v || "").toLowerCase().includes(query)));
        const table = el("table", { class: "grid" }, overgo.headerRow(["name", "source", "modality", "format", "files", "status"]));
        for (const d of rows) {
          table.appendChild(overgo.tableRow([el("button", { class: "link-button mono", text: d.name, onclick: () => showPreview(d.name) }),
            el("span", { class: "tag", text: d.source || "—" }), el("span", { text: d.modality || "—" }), el("span", { text: (d.formats || []).join(", ") || "—" }),
            String(d.files || 0), el("span", { class: d.available ? "" : "dim", text: d.available ? "available" : "missing" })]));
        }
        host.replaceChildren(el("div", { class: "note", text: rows.length + " shown" }), table);
      }
      search.addEventListener("input", render);
      render();
    },
  });
})();
