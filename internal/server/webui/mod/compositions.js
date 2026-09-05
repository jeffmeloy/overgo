(function () {
  "use strict";

  function id(overgo, value) {
    return overgo.el("span", { class: "mono", title: value || "", text: overgo.fmt.shortID(value || "-") });
  }

  function completion(overgo, state) {
    const labels = [
      ["contract-tested", state.contract_tested],
      ["runtime-wired", state.runtime_wired],
      ["cuda-verified", state.cuda_verified],
      ["evidence-promoted", state.evidence_promoted],
      ["production-active", state.production_active],
    ];
    return overgo.el("div", { class: "row" }, ...labels.map(([label, passed]) =>
      overgo.el("span", { class: "tag " + (passed ? "user_defined" : ""), text: (passed ? "✓ " : "○ ") + label })));
  }

  function graph(overgo, value) {
    if (!value) return overgo.el("div", { class: "note", text: "recipe graph unavailable" });
    return overgo.el("div", { class: "composition-graph" },
      id(overgo, value.source), overgo.el("span", { text: "capture" }), id(overgo, value.capture_contract),
      overgo.el("span", { class: "tag", text: value.operator }), id(overgo, value.weights),
      overgo.el("span", { text: "inject" }), id(overgo, value.injection_contract), id(overgo, value.target));
  }

  window.overgo.registerTab({
    id: "compositions",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      const status = el("span", { class: "note" });
      const host = el("div");
      const refresh = el("button", { class: "btn alt", text: "Refresh" });
      panel.append(
        el("div", { class: "section-title", text: "Composition authority" }),
        el("div", { class: "note", text: "Inventory, compatibility refusals, training controls, evaluation evidence, promotion history, active selection, and runtime evidence come from OvergoDB." }),
        el("div", { class: "row", style: "margin:12px 0" }, refresh, status), host);

      async function activate(recipe) {
        status.textContent = "activating " + fmt.shortID(recipe);
        try {
          await overgo.api.post("/compositions/activate", { recipe });
          await load();
        } catch (err) { status.textContent = overgo.friendlyError(err); }
      }

      async function generate(item, output) {
        status.textContent = "resolving promoted generation " + fmt.shortID(item.recipe);
        try {
          const value = await overgo.api.post("/compositions/generate", {
            source: item.source, target: item.target, task: item.task,
          });
          output.textContent = "recipe " + fmt.shortID(value.recipe) + " · plan " + fmt.shortID(value.plan) + " · output " + fmt.shortID(value.output);
          status.textContent = "promoted generation resolved";
        } catch (err) { status.textContent = overgo.friendlyError(err); }
      }

      function render(item) {
        const card = el("section", { class: "composition-card" });
		const generated = el("div", { class: "note" });
        card.append(
          el("div", { class: "row" },
            el("span", { class: "tag " + (item.compatible ? "user_defined" : "control"), text: item.compatible ? "compatible" : "refused" }),
            item.active ? el("span", { class: "tag user_defined", text: "active" }) : null,
            id(overgo, item.recipe),
			!item.active && item.compatible ? el("button", { class: "btn", text: "Activate", onclick: () => activate(item.recipe) }) : null,
			item.active && item.generation ? el("button", { class: "btn", text: "Open promoted generation", onclick: () => generate(item, generated) }) : null),
          item.refusal ? overgo.errorBanner(item.refusal) : null,
		  graph(overgo, item.graph), completion(overgo, item.completion || {}), generated);
        if (item.training) {
          card.append(el("div", { class: "section-title", text: "Bridge training controls and metrics" }),
            el("div", { class: "control-grid" },
              el("label", { class: "control" }, "Dataset", el("input", { class: "text", placeholder: "dataset artifact", disabled: true })),
              el("label", { class: "control" }, "Epochs", el("input", { class: "text", type: "number", min: "1", value: "1", disabled: true })),
              el("label", { class: "control" }, "Resume checkpoint", el("input", { class: "text", placeholder: "checkpoint artifact", disabled: true }))),
            el("div", { class: "note", text: "policy " + fmt.shortID(item.training.policy) + " · bridge-only optimizer · checkpoint/resume · " + item.training.metric_names.join(", ") }));
        }
        if (item.evaluation) {
          const value = item.evaluation;
          card.append(el("div", { class: "section-title", text: "Evaluation and promotion history" }),
            el("div", { class: "statgrid" },
              overgo.stat("Held-out gain", value.worst_held_out_gain),
              overgo.stat("Source dependence", value.worst_source_dependence),
              overgo.stat("Regression", value.worst_regression),
              overgo.stat("Seed spread", value.seed_spread),
              overgo.stat("Latency change", value.worst_latency_increase),
              overgo.stat("Device bytes", fmt.bytes(value.worst_device_byte_increase))),
            el("div", { class: "note", text: value.trial_count + " trials · evidence " + fmt.shortID(value.evidence) + " · policy " + fmt.shortID(value.policy) }));
        }
        if (item.runtime) {
          const runtime = item.runtime;
          card.append(el("div", { class: "section-title", text: "Runtime memory and latency evidence" }),
            el("div", { class: "statgrid" },
              overgo.stat("Plan", fmt.shortID(runtime.plan)),
              overgo.stat("Cache", fmt.shortID(runtime.cache_identity)),
              overgo.stat("Session bytes", fmt.bytes(runtime.sessions.artifact_bytes)),
              overgo.stat("Source residency", runtime.sessions.components[0].residency),
              overgo.stat("Injection residency", runtime.sessions.components[1].residency)));
        }
        return card;
      }

      async function load() {
        host.replaceChildren(el("div", { class: "note", text: "loading /compositions" }));
        try {
          const data = await overgo.api.get("/compositions");
          status.textContent = data.compositions.length + " composition recipes";
          host.replaceChildren(...data.compositions.map(render));
          if (!data.compositions.length) host.appendChild(el("div", { class: "note", text: "No composition recipes are published." }));
        } catch (err) { host.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }
      refresh.onclick = load;
      await load();
    },
  });
})();
