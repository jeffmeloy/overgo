(function () {
  "use strict";
  const refreshMilliseconds = 1000;
  let polling = null;

  function present(value, format) {
    return value == null ? "unknown" : String(format ? format(value) : value);
  }

  window.overgo.registerTab({
    id: "runtime",
    label: "Runtime",
    section: "inference",
    onActivate() { if (polling) polling.start(); },
    onDeactivate() { if (polling) polling.stop(); },
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      const identity = el("div", { class: "note" });
      const summary = el("div", { class: "statgrid" });
      const error = el("div");
      const slots = el("div");
      panel.append(el("div", { class: "section-title", text: "Runtime" }), identity, summary, error, slots);

      try {
        const info = await overgo.modelInfo();
        identity.textContent = info.model.name || info.model.id || "";
      } catch (err) {
        error.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
      }

      function render(data) {
        error.replaceChildren();
        summary.replaceChildren(
          overgo.stat("Active slots", data.filter((slot) => slot.is_processing).length),
          overgo.stat("Slots", data.length));
        const table = el("table", { class: "grid" });
        table.appendChild(el("tr", {},
          el("th", { text: "slot" }), el("th", { text: "task" }), el("th", { text: "state" }),
          el("th", { text: "context" }), el("th", { text: "prompt" }), el("th", { text: "cached" }),
          el("th", { text: "completion" }), el("th", { text: "prefill" }), el("th", { text: "decode" }),
          el("th", { text: "elapsed" })));
        for (const slot of data) {
          const timing = slot.timings || null;
          const elapsed = timing ? Number(timing.prompt_ms) + Number(timing.predicted_ms) : null;
          table.appendChild(el("tr", {},
            el("td", { class: "mono", text: slot.id }),
            el("td", { class: "mono", text: present(slot.id_task) }),
            el("td", {}, el("span", {
              class: "tag " + (slot.is_processing ? "user_defined" : ""),
              text: slot.is_processing ? "running" : "idle",
            })),
            el("td", { class: "mono", text: present(slot.n_ctx, fmt.grouped) }),
            el("td", { class: "mono", text: present(slot.n_prompt_tokens_processed, fmt.grouped) }),
            el("td", { class: "mono", text: present(slot.n_prompt_tokens_cache, fmt.grouped) }),
            el("td", { class: "mono", text: present(timing && timing.predicted_n, fmt.grouped) }),
            el("td", { class: "mono", text: present(timing && timing.prompt_per_second, (v) => Number(v).toFixed(2) + " tok/s") }),
            el("td", { class: "mono", text: present(timing && timing.predicted_per_second, (v) => Number(v).toFixed(2) + " tok/s") }),
            el("td", { class: "mono", text: present(elapsed, (v) => v.toFixed(2) + " ms") })));
        }
        slots.replaceChildren(table);
      }

      polling = overgo.poller(async (signal) => {
        try { render(await overgo.api.get("/slots", { signal })); }
        catch (err) { error.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }, refreshMilliseconds);
    },
  });
})();
