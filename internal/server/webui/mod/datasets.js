/* Datasets · active OvergoDB catalog. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "datasets",
    async mount(panel, overgo) {
      const { el } = overgo;
      const load = overgo.read(panel, async (signal) => {
        try {
          return await overgo.api.get("/datasets", { signal });
        } catch (err) {
          // A store with no dataset catalog yet is empty, not failing; a server without the catalog says so.
          if (err.status === 404) return null;
          if (err.status === 501) throw new Error("Dataset browsing is not configured on this server.");
          throw err;
        }
      }, render, { loading: "Loading datasets…" });

      function render(data) {
        // No catalog yet: say so, and open the Library's import, where a chat dataset file comes in.
        if (!data) {
          panel.replaceChildren(el("div", { class: "section-title", text: "Datasets" }),
            el("p", { class: "note", text: "No datasets are catalogued yet. Import a chat dataset file (alpaca, ShareGPT or OpenAI messages) and its conversations list here." }),
            el("button", { class: "btn alt", text: "Import a chat dataset", onclick: () => overgo.openLibrary("input[aria-label='Chat dataset file on disk']") }));
          return;
        }
        const search = el("input", { "aria-label": "Filter datasets", class: "text mw-440 my-s2", type: "search", placeholder: "filter by name / source / modality / format…", });
        const host = el("div");
        const preview = el("div");
        panel.replaceChildren(
          el("div", { class: "section-title", text: "Dataset registry" }),
          el("div", { class: "lens-summary" },
            el("span", { class: "note", text: data.count + " datasets" }),
            el("span", { class: "note mono", text: data.catalog })),
          search, host, preview);

        // previewText: a record in the OpenAI messages form reads as its conversation, any other text as written.
        function previewText(text) {
          let record = null;
          try { record = JSON.parse(text); } catch (_) { /* not JSON: shown as written */ }
          const messages = record && Array.isArray(record.messages) ? record.messages : null;
          if (!messages) return el("pre", { class: "preview-text", text });
          return el("div", { class: "dataset-conversation" }, ...messages.map((message) =>
            el("div", { class: "dataset-turn" }, el("strong", { text: message.role }), el("span", { class: "preview-text", text: message.content }))));
        }
        // The newest chosen dataset owns the preview: choosing another aborts the earlier read.
        let chosen = "";
        const showPreview = overgo.read(preview, (signal) => overgo.api.post("/datasets/preview", { name: chosen, position: 0, limit: 1 }, { signal }), (result) => {
          preview.replaceChildren(
            el("div", { class: "section-title", text: chosen }),
            ...result.examples.map((example) => el("div", { class: "dataset-example" },
              el("div", { class: "mono", text: example.id }),
              ...example.values.map((value) => el("div", {},
                el("span", { class: "tag", text: value.role + " / " + value.modality }),
                value.text ? previewText(value.text) :
                  el("span", { class: "note", text: value.encoding + " / " + value.bytes + " bytes" }))))));
        }, { loading: null });

        function renderRows() {
          const query = search.value.trim().toLowerCase();
          const rows = (data.datasets || []).filter((d) =>
            !query || [d.name, d.source, d.modality, ...(d.formats || [])].some((v) => (v || "").toLowerCase().includes(query)));
          const table = el("table", { class: "grid" }, overgo.headerRow(["name", "source", "modality", "format", "files", "status"]));
          for (const d of rows) {
            table.appendChild(overgo.tableRow([el("button", { class: "link-button mono", text: d.name, onclick: () => { chosen = d.name; showPreview(); } }),
              el("span", { class: "tag", text: d.source || "—" }), el("span", { text: d.modality || "—" }), el("span", { text: (d.formats || []).join(", ") || "—" }),
              String(d.files || 0), el("span", { class: d.available ? "" : "dim", text: d.available ? "available" : "missing" })]));
          }
          host.replaceChildren(el("div", { class: "note", text: rows.length + " shown" }), table);
        }
        search.addEventListener("input", renderRows);
        renderRows();
      }
      await load();
    },
  });
})();
