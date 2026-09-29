(function () {
  "use strict";
  window.overgo.registerTab({
    id: "recipe",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      // The bundles are an addendum: a failed bundle read shows the recipe without them.
      await overgo.read(panel, (signal) => Promise.all([
        overgo.api.get("/recipes/active?task=inference", { signal }),
        overgo.api.get("/capabilities/bundles", { signal }).then((bundleData) => bundleData.bundles || [], () => []),
      ]), render, { loading: "Loading the active recipe…" })();

      function render([data, bundles]) {
        clear(panel);
        if (!data.admitted) {
          panel.append(
            el("div", { class: "section-title", text: "Admission" }),
            overgo.errorBanner(data.refusal || "recipe refused"));
          return;
        }
        const runtime = data.recipe;
        const identity = runtime.identity;
        // The recipe reads as what it serves and how first; its identifiers wait in a fold, as Inspect keeps them.
        const served = overgo.capabilities();
        const name = (served && served.name) || fmt.shortID(identity.model);
        const declared = (value) => value || "not declared";
        const stages = runtime.stages.length;
        panel.appendChild(el("div", { class: "section-title", text: "Active recipe" }));
        const clauses = [stages + " compiled " + (stages === 1 ? "stage" : "stages"), identity.placement && "running on " + identity.placement,
          identity.residency && "weights held " + identity.residency].filter(Boolean);
        panel.appendChild(el("p", { class: "recipe-summary", text: "Serves " + name + " for " + declared(runtime.task) + ": " + clauses.join(", ") + "." }));
        panel.appendChild(el("div", { class: "statgrid" },
          overgo.stat("Model", name),
          overgo.stat("Task", declared(runtime.task)),
          overgo.stat("Runs on", declared(identity.placement)),
          overgo.stat("Weights held", declared(identity.residency)),
          overgo.stat("Runtime", declared(identity.runtime)),
          overgo.stat("Version", identity.recipe_version)));
        const identifiers = [overgo.stat("Recipe", fmt.shortID(identity.recipe)), overgo.stat("Model", fmt.shortID(identity.model)), overgo.stat("Cache", fmt.shortID(runtime.cache_identity))];
        const tags = (ids) => el("div", { class: "row" }, ...ids.map((id) => el("span", { class: "tag", title: id, text: fmt.shortID(id) })));
        panel.appendChild(overgo.fold("Identifiers", false, el("div", { class: "statgrid" }, ...identifiers),
          ...(runtime.evidence && runtime.evidence.length ? [el("div", { class: "note", text: "Admission evidence" }), tags(runtime.evidence)] : []),
          ...(bundles.length ? [el("div", { class: "note", text: "Capability bundles" }), tags(bundles)] : [])));

        panel.appendChild(el("div", { class: "section-title", text: "Compiled stages" }));
        panel.appendChild(overgo.table(["order", "node", "module", "placement", "residency", "session"],
          Array.from(runtime.stages.entries(), ([index, stage]) => [index, stage.node.id, stage.module.id, stage.node.placement, stage.node.residency || "", stage.node.session || ""])));

        // A recipe with no dependencies encodes its facts as null.
        const facts = runtime.required_facts || [];
        panel.appendChild(el("div", { class: "section-title", text: "Required facts" }));
        panel.appendChild(facts.length ? overgo.table(["role", "slot", "artifact"],
          facts.map((fact) => [fact.role, fact.slot || 0, el("span", { class: "mono", title: fact.artifact, text: fmt.shortID(fact.artifact) })]))
          : el("div", { class: "note", text: "This recipe requires no stored facts." }));
      }
    },
  });
})();
