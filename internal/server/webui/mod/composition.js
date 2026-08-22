(function () {
  "use strict";
  window.overgo.registerTab({
    id: "composition",
    label: "Composition",
    section: "workbench",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;

      async function render() {
        clear(panel);
        panel.appendChild(el("div", { class: "note", text: "loading composition lifecycle" }));
        let inventory;
        try {
          inventory = await overgo.api.get("/compositions");
        } catch (err) {
          panel.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
          return;
        }
        clear(panel);
        panel.appendChild(el("div", { class: "section-title", text: "Composition inventory" }));
        panel.appendChild(el("div", { class: "statgrid" },
          overgo.stat("Candidates", fmt.grouped(inventory.candidates.length)),
          overgo.stat("RepoDB sequence", fmt.grouped(inventory.sequence)),
          overgo.stat("Snapshot", fmt.shortID(inventory.head))));
        if (!inventory.candidates.length) {
          panel.appendChild(el("div", { class: "note", text: "No composition recipes are cataloged." }));
          return;
        }
        const active = inventory.candidates.find((candidate) => candidate.active);
        for (const candidate of inventory.candidates) {
          const card = el("section", { class: "card" });
          card.appendChild(el("div", { class: "row" },
            el("strong", { class: "mono", title: candidate.id, text: fmt.shortID(candidate.id) }),
            el("span", { class: "tag", text: candidate.active ? "active" : (candidate.compatible ? "compatible" : "refused") })));
          if (!candidate.compatible) {
            card.appendChild(overgo.errorBanner(candidate.refusal || "composition authority refused"));
            panel.appendChild(card);
            continue;
          }
          const recipe = candidate.recipe;
          const evidence = candidate.evidence;
          card.appendChild(el("div", { class: "statgrid" },
            overgo.stat("Source", fmt.shortID(recipe.source_model)),
            overgo.stat("Target", fmt.shortID(recipe.target_model)),
            overgo.stat("Task", recipe.task),
            overgo.stat("Operator", candidate.graph.operator),
            overgo.stat("Training policy", fmt.shortID(evidence.training_policy)),
            overgo.stat("Evaluation", fmt.shortID(evidence.evaluator)),
            overgo.stat("Promotion", fmt.shortID(evidence.promotion)),
            overgo.stat("Held-out split", fmt.shortID(evidence.held_out_split))));
          const stages = el("table", { class: "grid" });
          stages.appendChild(el("tr", {},
            el("th", { text: "component" }), el("th", { text: "placement" }),
            el("th", { text: "residency" }), el("th", { text: "lifetime" })));
          for (const component of (candidate.runtime && candidate.runtime.components) || []) {
            stages.appendChild(el("tr", {},
              el("td", { class: "mono", text: component.module }),
              el("td", { text: component.placement }),
              el("td", { text: component.residency }),
              el("td", { text: component.lifetime })));
          }
          if (candidate.runtime) card.appendChild(stages);
          if (!candidate.active) {
            const activate = el("button", { class: "primary", text: "Activate" });
            activate.addEventListener("click", async () => {
              activate.disabled = true;
              try {
                const request = { recipe: candidate.id };
                if (active) request.previous = active.id;
                await overgo.api.post("/compositions/activate", request);
                await render();
              } catch (err) {
                card.appendChild(overgo.errorBanner(overgo.friendlyError(err)));
                activate.disabled = false;
              }
            });
            card.appendChild(activate);
          }
          panel.appendChild(card);
        }
      }

      await render();
    },
  });
})();
