/* Datasets · browser. Scaffold: the Datasets section is present and navigable;
   its content is wired next to a read-only /datasets endpoint (dataset registry
   + row sampling), modeled on adaptive_new's dataset-registry API. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "datasets",
    label: "Datasets",
    section: "datasets",
    mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);
      panel.append(
        el("div", { class: "section-title", text: "Dataset browser" }),
        el("div", { class: "note", style: "max-width:60ch",
          text: "Read-only browser for the data root: a dataset registry (name, kind, modality, file and byte counts), per-dataset file listing, and row sampling — modeled on adaptive_new's dataset-registry API. Being wired to the server's /datasets endpoint." }));
    },
  });
})();
