/* Training · runs (read-only). Scaffold: the Training section is present and
   navigable; its content is wired next to a read-only /runs endpoint that
   browses run / recipe / evaluation artifacts from the RepoDB (no job control). */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "runs",
    label: "Runs",
    section: "training",
    mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);
      panel.append(
        el("div", { class: "section-title", text: "Training runs (read-only)" }),
        el("div", { class: "note", style: "max-width:60ch",
          text: "Browse training runs and their recipes, evaluations, and phase metrics from the RepoDB (via runrecord.ParseRun) — read-only, no job control. Being wired to the server's /runs endpoint." }));
    },
  });
})();
