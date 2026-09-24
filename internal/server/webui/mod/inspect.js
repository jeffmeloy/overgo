/* Inspect: the served model's analysis views in one place. Each view is a
   registered tab the manifest declares a view of Inspect; this tab lists them
   and mounts the chosen one, which ends with the next choice or the tab. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "inspect",
    mount(panel, overgo) {
      const { el } = overgo;
      const views = overgo.viewsOf("inspect");
      const host = el("div");
      const chooser = el("div", { class: "row view-chooser", role: "tablist", "aria-label": "Inspect views" },
        ...views.map((view) => el("button", { class: "chip", role: "tab", "data-view": view.id, text: view.label, onclick: () => show(view.id) })));
      panel.append(chooser, host);
      // The chosen view mounts, or says why it is refused and what enables it; a hash names the view it opens.
      function show(id) {
        const view = views.find((item) => item.id === id) || views.find((item) => item.enabled) || views[0];
        for (const button of chooser.children) button.setAttribute("aria-selected", String(button.dataset.view === view.id));
        history.replaceState(null, "", "#" + view.id);
        overgo.embed(view.id, host);
      }
      overgo.listen(window, "overgo-view", (event) => { if (event.detail.owner === "inspect") show(event.detail.view); });
      show(overgo.requestedView("inspect"));
    },
  });
})();
