(function () {
  "use strict";

  function present(value, format) { return value == null ? "—" : String(format ? format(value) : value); }

  window.overgo.registerTab({
    id: "runtime",
    async mount(panel, overgo) {
      const { el, fmt } = overgo;
      const identity = el("div", { class: "note" });
      const summary = el("div", { class: "statgrid" });
      const error = el("div");
      const slots = el("div");
      panel.append(el("div", { class: "section-title", text: "Runtime" }), identity, summary, error, slots);

      await overgo.read(error, () => overgo.modelInfo(), (info) => { identity.textContent = info.model.name || info.model.id || ""; }, { loading: null })();

      function render(data) {
        error.replaceChildren();
        const session = data.session;
        const authority = data.authority || null;
        if (authority) { identity.textContent = [authority.model, authority.recipe, authority.runtime, authority.residency].map(fmt.shortID).join("  "); }
        summary.replaceChildren(
          overgo.stat("Active sessions", session.active),
          overgo.stat("Available", session.available),
          overgo.stat("Capacity", session.capacity),
          overgo.stat("Waiting", session.waiting),
          overgo.stat("Loads", session.loads),
          overgo.stat("Reuses", session.reuses),
          overgo.stat("Evictions", session.evictions),
          overgo.stat("Retiring", session.retiring),
          overgo.stat("Recipe stages", authority ? authority.stages : "—"),
          overgo.stat("Evidence", authority ? (authority.evidence || []).length : "—"),
          overgo.stat("Device", session.device));
        const table = el("table", { class: "grid" }, overgo.headerRow(["slot", "task", "state", "context", "prompt", "cached", "completion", "prefill", "decode", "elapsed"]));
        for (const slot of data.slots) {
          const timing = slot.timings || null;
          const elapsed = timing ? Number(timing.prompt_ms) + Number(timing.predicted_ms) : null;
          table.appendChild(overgo.tableRow([slot.id, present(slot.id_task),
            el("span", { class: "tag " + (slot.is_processing ? "user_defined" : ""), text: slot.is_processing ? "running" : "idle" }),
            present(slot.n_ctx, fmt.grouped), present(slot.n_prompt_tokens_processed, fmt.grouped), present(slot.n_prompt_tokens_cache, fmt.grouped),
            present(timing && timing.predicted_n, fmt.grouped), present(timing && timing.prompt_per_second, (v) => Number(v).toFixed(2) + " tok/s"),
            present(timing && timing.predicted_per_second, (v) => Number(v).toFixed(2) + " tok/s"), present(elapsed, (v) => v.toFixed(2) + " ms")]));
        }
        slots.replaceChildren(table);
      }

      // The sessions view listens only while it shows.
      overgo.subscribe((name, value) => { if (name === "runtime.sessions") render(value); if (name === "stream.error") error.replaceChildren(overgo.failure(value)); }, { whileShown: true });
    },
  });

  window.overgo.registerTab({
    id: "activity",
    async mount(panel, overgo) {
      const { el, fmt } = overgo;
      const summary = el("div", { class: "statgrid" });
      const error = el("div");
      const operations = el("div");
      const rows = el("div");
      const workflow = el("div");
      const replay = el("div");
      panel.append(el("div", { class: "section-title", text: "Activity" }), el('button', { class: 'btn alt', text: 'Operations and results', onclick: () => overgo.showOperation('') }), summary, error, operations, rows, workflow, replay);
      const current = new Map();

      // A decision rides the strip's binding path (operations_shell.js): it names the request advertised now.
      function decide(operation, tool, answer) { overgo.decideOperation(operation, tool, answer).catch((err) => error.replaceChildren(overgo.failure(err))); }

      const dagHost = el("div");
      // The DAG view is the recipe graph joined with durable stage
      // receipts: each node badged by its lifecycle state, waiting nodes
      // pointing at the Inbox where their decision lives.
      let shownDAG = "";
      const showDAG = overgo.read(dagHost, (signal) => overgo.api.get("/operations/dag?id=" + encodeURIComponent(shownDAG), { signal }), (dag) => {
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
          el("div", { class: "section-title", text: "Workflow " + fmt.shortID(shownDAG) }), nodes, edges);
      }, { loading: "Loading the workflow…" });

      function renderOperations() {
        if (!current.size) {
          operations.replaceChildren(el("p", { class: "note empty-table", text: "No operations yet. Work you start (a generation, a training run, a validation) shows here while it runs, with its decisions." }), dagHost);
          return;
        }
        const table = el("table", { class: "grid" }, overgo.headerRow(["state", "task", "recipe", "progress", "attempts", "run", "outputs", "failure", "decision", "dag"]));
        for (const item of current.values()) {
          const progress = item.progress || {};
          const actions = item.recovery && item.recovery.actions || [];
          const decision = el("div", { class: "row" });
          for (const action of actions) decision.append(
            el("button", { class: "btn", text: "Grant " + action.code, onclick: () => decide(item.id, action.code, "grant") }),
            el("button", { class: "btn alt", text: "Decline", onclick: () => decide(item.id, action.code, "decline") }));
          table.appendChild(overgo.tableRow([el("span", { class: "tag " + (item.state === "completed" ? "user_defined" : "control"), text: item.state }),
            el("span", { text: item.task }), fmt.shortID(item.recipe),
            progress.total == null ? present(progress.completed) : progress.completed + " / " + progress.total,
            fmt.grouped((item.attempts || []).length), fmt.shortID(item.run), (item.outputs || []).map(fmt.shortID).join(", "),
            el("span", { text: item.failure || "" }), decision, el("button", { class: "btn alt", text: "DAG", onclick: () => { shownDAG = item.id; showDAG(); } })]));
        }
        operations.replaceChildren(table, dagHost);
      }

      let replayed = "";
      const showReplay = overgo.read(replay, (signal) => overgo.api.get("/interactions/replay?response=" + encodeURIComponent(replayed), { signal }), (value) => {
        // Media artifacts render as what they are (composer.js mediaPlayer); the rest stays in the raw trace view.
        const inline = (value.media || []).map((item) => {
          const src = overgo.contentURL(item.id);
          const kind = overgo.mediaKind(item.media_type || "");
          return kind === "document" ? el("a", { class: "mono", href: src, text: fmt.shortID(item.id) }) : overgo.mediaPlayer(kind, src, fmt.shortID(item.id));
        });
        replay.replaceChildren(
          ...(inline.length ? [el("div", { class: "row" }, ...inline)] : []),
          el("pre", { class: "mono", text: JSON.stringify(value, null, 2) }));
      }, { loading: "Loading the replay…" });

      function render(data) {
        error.replaceChildren();
        current.clear();
        for (const item of data.operations || []) current.set(item.id, item);
        renderOperations();
        summary.replaceChildren(
          overgo.stat("Observations", data.count),
          overgo.stat("Publish failures", data.publish_failures),
          overgo.stat("Catalog truncated", data.truncated ? "yes" : "no"));
        rows.replaceChildren(overgo.table(["started", "task", "outcome", "model", "recipe", "duration", "input", "output", "H2D", "D2H"], data.activity.map((item) => [
          el("span", { class: "nowrap", text: new Date(Number(item.started_unix_ns) / 1e6).toLocaleString(undefined, { dateStyle: "short", timeStyle: "medium" }) }),
          el("span", { text: item.task + (item.compatibility ? " / peer" : "") }),
          el("span", { class: "tag " + (item.outcome === "succeeded" ? "user_defined" : "control"), text: item.outcome }),
          fmt.shortID(item.model), fmt.shortID(item.recipe), (Number(item.measured_ns) / 1e6).toFixed(2) + " ms",
          fmt.grouped(item.usage.input_tokens || 0), fmt.grouped(item.usage.output_tokens || 0),
          fmt.bytes(item.resources.host_to_device_bytes || 0), fmt.bytes(item.resources.device_to_host_bytes || 0)])));

        const stageTable = overgo.table(["stage", "state", "attempt", "operation", "failure"], (data.stages || []).map((stage) => [
          stage.node, el("span", { text: stage.state }), stage.attempt, fmt.shortID(stage.operation), el("span", { text: stage.failure || "" })]));
        const decisionTable = overgo.table(["answer", "tool", "operation", "request"], (data.decisions || []).map((decision) => [
          decision.answer, el("span", { text: decision.tool }), fmt.shortID(decision.operation), fmt.shortID(decision.request)]));
        const interactionTable = el("table", { class: "grid" }, overgo.headerRow(["response", "node", "trace", "action"]));
        for (const interaction of data.interactions || []) interactionTable.appendChild(el("tr", {},
          el("td", { class: "mono", text: interaction.response }), el("td", { class: "mono", text: interaction.node }),
          el("td", { class: "mono", text: fmt.shortID(interaction.trace) }), el("td", {},
            el("button", { class: "btn alt", text: "Replay", onclick: () => { replayed = interaction.response; showReplay(); } }))));
        workflow.replaceChildren(
          overgo.fold("Workflow stages", false, stageTable),
          overgo.fold("Tool decisions", false, decisionTable),
          overgo.fold("Interaction replay", true, interactionTable));
      }

      // The activity view listens only while it shows.
      overgo.subscribe((name, value) => {
        if (name === "runtime.activity") render(value);
        if (name === "operation.snapshot") { current.clear(); for (const item of value) current.set(item.id, item); renderOperations(); }
        if (name === "operation") { current.set(value.status.id, value.status); renderOperations(); }
        if (name === "stream.error") error.replaceChildren(overgo.failure(value));
      }, { whileShown: true });
    },
  });
})();
