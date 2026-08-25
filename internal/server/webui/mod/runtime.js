(function () {
  "use strict";
  let runtimeView = null;
  let runtimeUnsubscribe = null;
  let activityView = null;
  let activityUnsubscribe = null;

  function present(value, format) {
    return value == null ? "unknown" : String(format ? format(value) : value);
  }

  function activateRuntime() {
    if (!runtimeUnsubscribe && runtimeView) runtimeUnsubscribe = window.overgo.runtimeEvents.subscribe(runtimeView);
  }

  function deactivateRuntime() {
    if (runtimeUnsubscribe) runtimeUnsubscribe();
    runtimeUnsubscribe = null;
  }

  function activateActivity() {
    if (!activityUnsubscribe && activityView) activityUnsubscribe = window.overgo.runtimeEvents.subscribe(activityView);
  }

  function deactivateActivity() {
    if (activityUnsubscribe) activityUnsubscribe();
    activityUnsubscribe = null;
  }

  window.overgo.registerTab({
    id: "runtime",
    onActivate: activateRuntime,
    onDeactivate: deactivateRuntime,
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

      runtimeView = (name, value) => {
        if (name === "runtime.sessions") render(value);
        if (name === "stream.error") error.replaceChildren(overgo.errorBanner(overgo.friendlyError(value)));
      };
      activateRuntime();
    },
  });

  window.overgo.registerTab({
    id: "activity",
    onActivate: activateActivity,
    onDeactivate: deactivateActivity,
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      const summary = el("div", { class: "statgrid" });
      const error = el("div");
	  const operations = el("div");
      const rows = el("div");
	  const workflow = el("div");
	  const replay = el("div");
	  panel.append(el("div", { class: "section-title", text: "Activity" }), summary, error, operations, rows, workflow, replay);
	  const current = new Map();

	  async function decide(operation, tool, answer) {
		try {
		  await overgo.api.post("/operations/decision", { operation, tool, answer });
		} catch (err) {
		  error.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
		}
	  }

	  const dagHost = el("div");
	  // The DAG view is the recipe graph joined with durable stage
	  // receipts: each node badged by its lifecycle state, waiting nodes
	  // pointing at the Inbox where their decision lives.
	  async function renderDAG(operation) {
		try {
		  const dag = await overgo.api.get("/operations/dag?id=" + encodeURIComponent(operation));
		  const nodes = el("div", { class: "row" });
		  for (const node of dag.nodes || []) {
			const stateClass = node.state === "failed" ? "tag tag-danger"
			  : node.state === "completed" ? "tag user_defined" : "tag control";
			nodes.append(el("div", { class: "card" },
			  el("div", { class: "mono", text: node.id }),
			  el("div", { class: "note", text: node.module }),
			  el("span", { class: stateClass, text: node.state + (node.failure ? " / " + node.failure : "") }),
			  node.state === "waiting" ? el("div", { class: "note", text: "decision waits in the Inbox" }) : ""));
		  }
		  const edges = el("div", { class: "note" },
			(dag.edges || []).map((edge) => edge.from + " → " + edge.to).join("   "));
		  dagHost.replaceChildren(
			el("div", { class: "section-title", text: "Workflow " + fmt.shortID(operation) }), nodes, edges);
		} catch (err) {
		  error.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
		}
	  }

	  function renderOperations() {
		const table = el("table", { class: "grid" });
		table.appendChild(el("tr", {},
		  el("th", { text: "state" }), el("th", { text: "task" }), el("th", { text: "recipe" }),
		  el("th", { text: "progress" }), el("th", { text: "attempts" }), el("th", { text: "run" }),
		  el("th", { text: "outputs" }), el("th", { text: "failure" }), el("th", { text: "decision" }), el("th", { text: "dag" })));
		for (const item of current.values()) {
		  const progress = item.progress || {};
		  const actions = item.recovery && item.recovery.actions || [];
		  const decision = el("div", { class: "row" });
		  for (const action of actions) decision.append(
			el("button", { class: "btn", text: "Grant " + action.code, onclick: () => decide(item.id, action.code, "grant") }),
			el("button", { class: "btn alt", text: "Decline", onclick: () => decide(item.id, action.code, "decline") }));
		  table.appendChild(el("tr", {},
			el("td", {}, el("span", { class: "tag " + (item.state === "completed" ? "user_defined" : "control"), text: item.state })),
			el("td", { text: item.task }), el("td", { class: "mono", text: fmt.shortID(item.recipe) }),
			el("td", { class: "mono", text: progress.total == null ? present(progress.completed) : progress.completed + " / " + progress.total }),
			el("td", { class: "mono", text: fmt.grouped((item.attempts || []).length) }),
			el("td", { class: "mono", text: fmt.shortID(item.run) }),
			el("td", { class: "mono", text: (item.outputs || []).map(fmt.shortID).join(", ") }),
			el("td", { text: item.failure || "" }), el("td", {}, decision),
			el("td", {}, el("button", { class: "btn alt", text: "DAG", onclick: () => renderDAG(item.id) }))));
		}
		operations.replaceChildren(table, dagHost);
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
			el("td", { text: item.task + (item.compatibility ? " / peer" : "") }),
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

		const stageTable = el("table", { class: "grid" });
		stageTable.appendChild(el("tr", {}, el("th", { text: "stage" }), el("th", { text: "state" }),
		  el("th", { text: "attempt" }), el("th", { text: "operation" }), el("th", { text: "failure" })));
		for (const stage of data.stages || []) stageTable.appendChild(el("tr", {},
		  el("td", { class: "mono", text: stage.node }), el("td", { text: stage.state }),
		  el("td", { class: "mono", text: stage.attempt }), el("td", { class: "mono", text: fmt.shortID(stage.operation) }),
		  el("td", { text: stage.failure || "" })));
		const decisionTable = el("table", { class: "grid" });
		decisionTable.appendChild(el("tr", {}, el("th", { text: "answer" }), el("th", { text: "tool" }),
		  el("th", { text: "operation" }), el("th", { text: "request" })));
		for (const decision of data.decisions || []) decisionTable.appendChild(el("tr", {},
		  el("td", { text: decision.answer }), el("td", { text: decision.tool }),
		  el("td", { class: "mono", text: fmt.shortID(decision.operation) }),
		  el("td", { class: "mono", text: fmt.shortID(decision.request) })));
		const interactionTable = el("table", { class: "grid" });
		interactionTable.appendChild(el("tr", {}, el("th", { text: "response" }), el("th", { text: "node" }),
		  el("th", { text: "trace" }), el("th", { text: "action" })));
		for (const interaction of data.interactions || []) interactionTable.appendChild(el("tr", {},
		  el("td", { class: "mono", text: interaction.response }), el("td", { class: "mono", text: interaction.node }),
		  el("td", { class: "mono", text: fmt.shortID(interaction.trace) }), el("td", {},
			el("button", { class: "btn alt", text: "Replay", onclick: async () => {
			  try {
				const value = await overgo.api.get("/interactions/replay?response=" + encodeURIComponent(interaction.response));
				// Media artifacts render as what they are -- images, video,
				// audio over the artifact content route -- never as opaque
				// identities; everything else stays in the raw trace view.
				const inline = (value.media || []).map((item) => {
				  const src = "/artifacts/content?id=" + encodeURIComponent(item.id);
				  const type = item.media_type || "";
				  if (type.startsWith("image/")) return el("img", { src, style: "max-width:320px;max-height:240px" });
				  if (type.startsWith("video/")) return el("video", { src, controls: "", style: "max-width:420px" });
				  if (type.startsWith("audio/")) return el("audio", { src, controls: "" });
				  return el("a", { class: "mono", href: src, text: fmt.shortID(item.id) });
				});
				replay.replaceChildren(
				  ...(inline.length ? [el("div", { class: "row" }, ...inline)] : []),
				  el("pre", { class: "mono", text: JSON.stringify(value, null, 2) }));
			  } catch (err) {
				replay.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
			  }
			} }))));
		workflow.replaceChildren(
		  overgo.fold("Workflow stages", false, stageTable),
		  overgo.fold("Tool decisions", false, decisionTable),
		  overgo.fold("Interaction replay", true, interactionTable));
      }

	  activityView = (name, value) => {
		if (name === "runtime.activity") render(value);
		if (name === "operation.snapshot") {
		  current.clear();
		  for (const item of value) current.set(item.id, item);
		  renderOperations();
		}
		if (name === "operation") {
		  current.set(value.status.id, value.status);
		  renderOperations();
		}
		if (name === "stream.error") error.replaceChildren(overgo.errorBanner(overgo.friendlyError(value)));
	  };
	  activateActivity();
    },
  });
})();
