/* Training · runs (read-only). Browses training/run artifacts from the server's
   /runs endpoint (RepoDB → runrecord.ParseRun): outcome, recipe, code commit,
   wall time, and the heaviest phases. Paged. No job control. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "runs",
    label: "Runs",
    section: "training",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      const limit = 50;
      let offset = 0;

      clear(panel);
      const status = el("span", { class: "note" });
      const prev = el("button", { class: "btn alt", onclick: () => { offset = Math.max(0, offset - limit); load(); } }, "‹ prev");
      const next = el("button", { class: "btn alt", onclick: () => { offset += limit; load(); } }, "next ›");
      panel.append(
        el("div", { class: "section-title", text: "Training runs (read-only)" }),
        el("div", { class: "row", style: "margin-bottom:10px" }, prev, next, status));
      const host = el("div");
      panel.appendChild(host);

      function outcomeClass(outcome) {
        return outcome === "succeeded" ? "user_defined" : (outcome === "failed" ? "control" : "");
      }
      async function load() {
        host.replaceChildren(el("div", { class: "note", text: "loading /runs…" }));
        let data;
        try {
          data = await overgo.api.get("/runs?offset=" + offset + "&limit=" + limit);
        } catch (err) {
          const message = err.status === 501
            ? "Run browsing is not configured on this server (no RepoDB store)."
            : overgo.friendlyError(err);
          host.replaceChildren(overgo.errorBanner(message));
          return;
        }
        const first = data.count === 0 ? 0 : data.offset + 1;
        const last = data.offset + data.runs.length;
        status.textContent = first + "–" + last + " of " + fmt.grouped(data.count) + " runs";
        prev.disabled = data.offset === 0;
        next.disabled = last >= data.count;

        const table = el("table", { class: "grid" });
        table.appendChild(el("tr", {},
          el("th", { text: "outcome" }), el("th", { text: "recipe" }), el("th", { text: "commit" }),
          el("th", { text: "wall" }), el("th", { text: "heaviest phases" }), el("th", { text: "in/out" })));
        for (const run of data.runs) {
          const phases = [...(run.phases || [])].sort((a, b) => b.ms - a.ms).slice(0, 3)
            .map((p) => p.phase + " " + p.ms.toFixed(0) + "ms").join(", ");
          table.appendChild(el("tr", {},
            el("td", {}, el("span", { class: "tag " + outcomeClass(run.outcome), text: run.outcome })),
            el("td", { class: "mono", title: run.recipe, text: fmt.shortID(run.recipe) }),
            el("td", { class: "mono", text: (run.code_commit || "").slice(0, 10) || "—" }),
            el("td", { class: "mono", text: run.measured_ms ? run.measured_ms.toFixed(0) + " ms" : "—" }),
            el("td", { class: "dim", text: phases || "—" }),
            el("td", { class: "mono", text: run.inputs + "/" + run.outputs })));
        }
        host.replaceChildren(table);
      }
      await load();
    },
  });
})();
