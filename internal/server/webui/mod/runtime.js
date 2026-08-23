(function () {
  "use strict";
  const refreshMilliseconds = 1000;
  let runtimePolling = null;
  let activityStream = null;

  function present(value, format) {
    return value == null ? "unknown" : String(format ? format(value) : value);
  }

  window.overgo.registerTab({
    id: "runtime",
    label: "Runtime",
    section: "inference",
    onActivate() { if (runtimePolling) runtimePolling.start(); },
    onDeactivate() { if (runtimePolling) runtimePolling.stop(); },
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
        const session = data.session;
		const authority = data.authority || null;
		if (authority) {
			identity.textContent = [authority.model, authority.recipe,
				authority.runtime, authority.residency].map(fmt.shortID).join("  ");
		}
        summary.replaceChildren(
          overgo.stat("Active sessions", session.active),
          overgo.stat("Available", session.available),
          overgo.stat("Capacity", session.capacity),
		  overgo.stat("Waiting", session.waiting),
		  overgo.stat("Loads", session.loads),
		  overgo.stat("Reuses", session.reuses),
		  overgo.stat("Evictions", session.evictions),
		  overgo.stat("Retiring", session.retiring),
		  overgo.stat("Recipe stages", authority ? authority.stages : "unknown"),
		  overgo.stat("Evidence", authority ? (authority.evidence || []).length : "unknown"),
		  overgo.stat("Device", session.device));
        const table = el("table", { class: "grid" });
        table.appendChild(el("tr", {},
          el("th", { text: "slot" }), el("th", { text: "task" }), el("th", { text: "state" }),
          el("th", { text: "context" }), el("th", { text: "prompt" }), el("th", { text: "cached" }),
          el("th", { text: "completion" }), el("th", { text: "prefill" }), el("th", { text: "decode" }),
          el("th", { text: "elapsed" })));
        for (const slot of data.slots) {
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

      runtimePolling = overgo.poller(async (signal) => {
        try { render(await overgo.api.get("/runtime/sessions", { signal })); }
        catch (err) { error.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }, refreshMilliseconds);
    },
  });

  window.overgo.registerTab({
    id: "activity",
    label: "Activity",
    section: "inference",
    onActivate() { if (activityStream) activityStream.start(); },
    onDeactivate() { if (activityStream) activityStream.stop(); },
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      const summary = el("div", { class: "statgrid" });
      const error = el("div");
	  const operations = el("div");
      const rows = el("div");
	  panel.append(el("div", { class: "section-title", text: "Activity" }), summary, error, operations, rows);
	  const current = new Map();

	  function renderOperations() {
		const table = el("table", { class: "grid" });
		table.appendChild(el("tr", {},
		  el("th", { text: "state" }), el("th", { text: "task" }), el("th", { text: "recipe" }),
		  el("th", { text: "progress" }), el("th", { text: "attempts" }), el("th", { text: "run" }),
		  el("th", { text: "outputs" }), el("th", { text: "failure" })));
		for (const item of current.values()) {
		  const progress = item.progress || {};
		  table.appendChild(el("tr", {},
			el("td", {}, el("span", { class: "tag " + (item.state === "completed" ? "user_defined" : "control"), text: item.state })),
			el("td", { text: item.task }), el("td", { class: "mono", text: fmt.shortID(item.recipe) }),
			el("td", { class: "mono", text: progress.total == null ? present(progress.completed) : progress.completed + " / " + progress.total }),
			el("td", { class: "mono", text: fmt.grouped((item.attempts || []).length) }),
			el("td", { class: "mono", text: fmt.shortID(item.run) }),
			el("td", { class: "mono", text: (item.outputs || []).map(fmt.shortID).join(", ") }),
			el("td", { text: item.failure || "" })));
		}
		operations.replaceChildren(table);
	  }

      function render(data) {
        error.replaceChildren();
		current.clear();
		for (const item of data.operations || []) current.set(item.id, item);
		renderOperations();
        summary.replaceChildren(
          overgo.stat("Observations", data.count),
          overgo.stat("Publish failures", data.publish_failures),
          overgo.stat("Catalog truncated", data.truncated ? "yes" : "no"));
        const table = el("table", { class: "grid" });
        table.appendChild(el("tr", {},
          el("th", { text: "started" }), el("th", { text: "task" }), el("th", { text: "outcome" }),
          el("th", { text: "model" }), el("th", { text: "recipe" }), el("th", { text: "duration" }),
          el("th", { text: "input" }), el("th", { text: "output" }), el("th", { text: "H2D" }), el("th", { text: "D2H" })));
        for (const item of data.activity) {
          table.appendChild(el("tr", {},
            el("td", { class: "mono", text: new Date(Number(item.started_unix_ns) / 1e6).toLocaleString() }),
            el("td", { text: item.task }),
            el("td", {}, el("span", { class: "tag " + (item.outcome === "succeeded" ? "user_defined" : "control"), text: item.outcome })),
            el("td", { class: "mono", text: fmt.shortID(item.model) }),
            el("td", { class: "mono", text: fmt.shortID(item.recipe) }),
            el("td", { class: "mono", text: (Number(item.measured_ns) / 1e6).toFixed(2) + " ms" }),
            el("td", { class: "mono", text: fmt.grouped(item.usage.input_tokens || 0) }),
            el("td", { class: "mono", text: fmt.grouped(item.usage.output_tokens || 0) }),
            el("td", { class: "mono", text: fmt.bytes(item.resources.host_to_device_bytes || 0) }),
            el("td", { class: "mono", text: fmt.bytes(item.resources.device_to_host_bytes || 0) })));
        }
        rows.replaceChildren(table);
      }

	  let active = false;
	  let controller = null;
	  async function refresh(signal) { render(await overgo.api.get("/runtime/activity", { signal })); }
	  async function connect(signal) {
		await overgo.api.events("/runtime/activity/stream", (name, value) => {
		  if (name === "operation.snapshot") {
			current.clear();
			for (const item of value) current.set(item.id, item);
		  } else if (name === "operation") {
			current.set(value.status.id, value.status);
			if (["completed", "blocked", "cancelled", "failed"].includes(value.status.state)) refresh(signal).catch(() => {});
		  }
		  renderOperations();
		}, { signal });
	  }
	  activityStream = {
		start() {
		  if (active) return;
		  active = true;
		  controller = new AbortController();
		  refresh(controller.signal).then(() => connect(controller.signal)).catch((err) => {
			if (err.name !== "AbortError") error.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
		  });
		},
		stop() {
		  active = false;
		  if (controller) controller.abort();
		  controller = null;
		},
	  };
    },
  });
})();
