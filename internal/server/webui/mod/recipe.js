(function () {
  "use strict";
  window.overgo.registerTab({
    id: "recipe",
    label: "Recipe",
    section: "workbench",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      panel.appendChild(el("div", { class: "note", text: "loading active recipe" }));

      let data;
      let bundles = [];
      try {
        data = await overgo.api.get("/recipes/active?task=inference");
      } catch (err) {
        panel.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
        return;
      }
	  try {
		const bundleData = await overgo.api.get("/capabilities/bundles");
		bundles = bundleData.bundles || [];
	  } catch (_) {
		bundles = [];
	  }
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
      const stages = el("table", { class: "grid" });
      stages.appendChild(el("tr", {},
        el("th", { text: "order" }), el("th", { text: "node" }), el("th", { text: "module" }),
        el("th", { text: "placement" }), el("th", { text: "residency" }), el("th", { text: "session" })));
      for (const [index, stage] of runtime.stages.entries()) {
        stages.appendChild(el("tr", {},
          el("td", { class: "mono", text: index }),
          el("td", { class: "mono", text: stage.node.id }),
          el("td", { class: "mono", text: stage.module.id }),
          el("td", { text: stage.node.placement }),
          el("td", { text: stage.node.residency || "" }),
          el("td", { text: stage.node.session || "" })));
      }
      panel.appendChild(stages);

      panel.appendChild(el("div", { class: "section-title", text: "Required facts" }));
      const facts = el("table", { class: "grid" });
      facts.appendChild(el("tr", {},
        el("th", { text: "role" }), el("th", { text: "slot" }), el("th", { text: "artifact" })));
      for (const fact of runtime.required_facts) {
        facts.appendChild(el("tr", {},
          el("td", { text: fact.role }),
          el("td", { class: "mono", text: fact.slot || 0 }),
          el("td", { class: "mono", title: fact.artifact, text: fmt.shortID(fact.artifact) })));
      }
      panel.appendChild(facts);

      if (runtime.evidence && runtime.evidence.length) {
        panel.appendChild(el("div", { class: "section-title", text: "Admission evidence" }));
        const evidence = el("div", { class: "row" });
        for (const id of runtime.evidence) {
          evidence.appendChild(el("span", { class: "tag", title: id, text: fmt.shortID(id) }));
        }
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
