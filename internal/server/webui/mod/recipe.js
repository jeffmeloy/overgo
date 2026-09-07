(function () {
  "use strict";
  window.overgo.registerTab({
    id: "recipe",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      panel.appendChild(el("div", { class: "note", text: "loading active recipe" }));

      let data;
      let bundles = [];
      try {
        data = await overgo.api.get("/recipes/active?task=inference");
      } catch (err) { panel.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); return; }
	  try {
		const bundleData = await overgo.api.get("/capabilities/bundles");
		bundles = bundleData.bundles || [];
	  } catch (_) { bundles = []; }
      clear(panel);
      if (!data.admitted) {
        panel.append(
          el("div", { class: "section-title", text: "Admission" }),
          overgo.errorBanner(data.refusal || "recipe refused"));
        return;
      }
      const runtime = data.recipe;
      const identity = runtime.identity;
      panel.appendChild(el("div", { class: "section-title", text: "Active recipe" }));
      panel.appendChild(el("div", { class: "statgrid" },
        overgo.stat("Recipe", fmt.shortID(identity.recipe)),
        overgo.stat("Version", identity.recipe_version),
        overgo.stat("Runtime", identity.runtime),
        overgo.stat("Placement", identity.placement),
        overgo.stat("Residency", identity.residency),
        overgo.stat("Cache", fmt.shortID(runtime.cache_identity))));

      panel.appendChild(el("div", { class: "section-title", text: "Compiled stages" }));
      panel.appendChild(overgo.table(["order", "node", "module", "placement", "residency", "session"],
        Array.from(runtime.stages.entries(), ([index, stage]) => [index, stage.node.id, stage.module.id, stage.node.placement, stage.node.residency || "", stage.node.session || ""])));

      panel.appendChild(el("div", { class: "section-title", text: "Required facts" }));
      panel.appendChild(overgo.table(["role", "slot", "artifact"],
        runtime.required_facts.map((fact) => [fact.role, fact.slot || 0, el("span", { class: "mono", title: fact.artifact, text: fmt.shortID(fact.artifact) })])));

      if (runtime.evidence && runtime.evidence.length) {
        panel.appendChild(el("div", { class: "section-title", text: "Admission evidence" }));
        const evidence = el("div", { class: "row" });
        for (const id of runtime.evidence) evidence.appendChild(el("span", { class: "tag", title: id, text: fmt.shortID(id) }));
        panel.appendChild(evidence);
      }
	  if (bundles.length) {
		panel.appendChild(el("div", { class: "section-title", text: "Capability bundles" }));
		const bundleList = el("div", { class: "row" });
		for (const id of bundles) bundleList.appendChild(el("span", { class: "tag", title: id, text: fmt.shortID(id) }));
		panel.appendChild(bundleList);
	  }
    },
  });
})();
