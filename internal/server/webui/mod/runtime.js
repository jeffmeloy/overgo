(function () {
  "use strict";

  function present(value, format) { return value == null ? "—" : String(format ? format(value) : value); }
  // measure: a quantity and its unit held on one line.
  function measure(text) { return window.overgo.el("span", { class: "mono nowrap", text }); }

  // The monitor reads /metrics once a second while the tab shows, and each trend keeps a minute of reads.
  const monitorMS = 1000, monitorPoints = 60;
  // metricValues: the samples a /metrics answer holds, by name; a line that opens with # describes one.
  function metricValues(text) {
    const values = new Map();
    for (const line of text.split("\n")) {
      const space = line.lastIndexOf(" ");
      if (space > 0 && line[0] !== "#") values.set(line.slice(0, space), Number(line.slice(space + 1)));
    }
    return values;
  }

  window.overgo.registerTab({
    id: "runtime",
    async mount(panel, overgo) {
      const { el, fmt } = overgo;
      const identity = el("div", { class: "note" });
      const summary = el("div", { class: "statgrid" });
      const error = el("div");
      const slots = el("div");
      const trend = el("div");
      const loop = el("section", { class: "loop-status", role: "region", "aria-label": "unattended loop", hidden: true });
      const monitorStats = el("div", { class: "statgrid" }), monitorTrend = el("div", { class: "turn-trend" });
      const monitor = el("section", { class: "monitor", role: "region", "aria-label": "monitor", hidden: true },
        el("div", { class: "section-title", text: "Monitor" }), monitorStats, monitorTrend);
      panel.append(el("div", { class: "section-title", text: "Runtime" }), loop, identity, monitor,
        el("div", { class: "section-title", text: "Sessions" }), summary, error, slots, trend);

      // The monitor draws the served card and its experts live. A rate is a counter's change between two reads:
      // decode over the time between them, prefill over the time the prompts took, so a read with no prompt keeps
      // the last; hits are the routed experts found on the card between two reads, or since the model loaded.
      // A /metrics without a card's readings or experts draws none, and one that fails hides the monitor.
      let previous = null, readAt = 0, decode = null, prefill = null, hits = null, reading = false, frame = 0;
      const trends = new Map();
      overgo.onRelease(() => cancelAnimationFrame(frame));
      async function sample() {
        if (reading || document.hidden) return;
        reading = true;
        try {
          const values = metricValues(await (await overgo.api.stream("/metrics", null, { method: "GET", signal: overgo.signal })).text());
          const now = performance.now();
          // A new server process starts its counters again.
          if (previous && values.get("overgo_process_uptime_seconds") < previous.get("overgo_process_uptime_seconds")) previous = null;
          const change = (name) => previous ? values.get(name) - previous.get(name) : 0;
          decode = previous ? change("overgo_generated_tokens_total") / ((now - readAt) / 1000) : null;
          if (change("overgo_prompt_seconds_total") > 0) prefill = change("overgo_prompt_tokens_total") / change("overgo_prompt_seconds_total");
          if (change("overgo_expert_routes_total") > 0) hits = change("overgo_expert_routes_card_total") / change("overgo_expert_routes_total");
          else if (hits == null && values.get("overgo_expert_routes_total")) hits = values.get("overgo_expert_routes_card_total") / values.get("overgo_expert_routes_total");
          previous = values;
          readAt = now;
          // Draws share the page's frame: a read that lands while one is pending draws in it.
          if (!frame) frame = requestAnimationFrame(() => { frame = 0; drawMonitor(previous); });
        } catch (err) {
          if (!overgo.signal.aborted) monitor.hidden = true;
        } finally { reading = false; }
      }
      function drawMonitor(values) {
        const has = (name) => values.has(name), get = (name) => values.get(name);
        const used = get("overgo_device_memory_total_bytes") - get("overgo_device_memory_free_bytes");
        const rate = (value) => value == null ? ["—", ""] : [value.toFixed(1), "tok/s"];
        const shown = [
          ["Decode", ...rate(decode), decode],
          ["Prefill", ...rate(prefill)],
          has("overgo_card_utilization_percent") && ["GPU load", get("overgo_card_utilization_percent"), "%", get("overgo_card_utilization_percent")],
          has("overgo_device_memory_total_bytes") && ["Card memory", fmt.bytes(used), "of " + fmt.bytes(get("overgo_device_memory_total_bytes")), used],
          has("overgo_card_temperature_celsius") && ["Temperature", get("overgo_card_temperature_celsius"), "°C"],
          has("overgo_card_power_watts") && ["Power", Math.round(get("overgo_card_power_watts")), has("overgo_card_power_limit_watts") ? "W of " + Math.round(get("overgo_card_power_limit_watts")) + " W" : "W"],
          has("overgo_card_pcie_generation") && ["PCIe", "Gen" + get("overgo_card_pcie_generation") + (has("overgo_card_pcie_width") ? " x" + get("overgo_card_pcie_width") : ""),
            has("overgo_card_pcie_received_bytes_per_second") ? fmt.bytes(get("overgo_card_pcie_received_bytes_per_second")) + "/s to the card" : ""],
          has("overgo_experts_card") && ["Experts on the card", fmt.grouped(get("overgo_experts_card")), "in " + fmt.bytes(get("overgo_experts_card_bytes"))],
          has("overgo_experts_host") && ["Experts in RAM", fmt.grouped(get("overgo_experts_host")), "in " + fmt.bytes(get("overgo_experts_host_bytes"))],
          has("overgo_experts_card") && ["Expert hits", hits == null ? "—" : Math.round(100 * hits), hits == null ? "" : "% on the card", hits == null ? null : 100 * hits],
        ].filter(Boolean);
        monitor.hidden = false;
        monitorStats.replaceChildren(...shown.map(([label, value, unit]) => overgo.stat(label, value, unit)));
        // A measure with a trend keeps its last reads; the card's memory reads against its total, the shares against all.
        const ceilings = { "GPU load": 100, "Card memory": get("overgo_device_memory_total_bytes"), "Expert hits": 100 };
        for (const [label, , , point] of shown) if (point != null) trends.set(label, [...(trends.get(label) || []), point].slice(-monitorPoints));
        const drawn = shown.map(([label]) => [label, trends.get(label) || []]).filter(([, points]) => points.length > 1);
        monitorTrend.replaceChildren(...drawn.map(([label, points]) => el("div", { class: "turn-trend-line" },
          el("div", { class: "note", text: label + " over the last " + points.length + " reads" }),
          window.overgo.viz.sparkline(points, { height: 36, label, max: ceilings[label] }))));
      }
      overgo.whileActive(() => {
        previous = null;
        sample();
        const timer = setInterval(sample, monitorMS);
        return () => clearInterval(timer);
      });

      // The unattended loop of the checkout the server runs from: whether it runs, the row it works, its
      // last landing, and a stop it honours. A server that runs from no such checkout shows none.
      function renderLoop(status, failed) {
        loop.hidden = false;
        const state = status.stopped ? "stopped" : status.running ? "running" : "not running";
        const stop = el("button", { class: "btn alt", text: "Stop the loop", disabled: status.stopped, onclick: async () => {
          stop.disabled = true;
          loopView++;
          try { renderLoop(await overgo.api.post("/loop/stop", {}, { signal: overgo.signal })); }
          catch (err) { if (!overgo.signal.aborted) renderLoop(status, err); }
        } });
        const landing = status.landing || {};
        loop.replaceChildren(...[
          el("div", { class: "row" }, el("h3", { text: "Unattended loop" }), el("span", { class: "tag loop-state" + (state === "running" ? " ok" : state === "stopped" ? " caution" : ""), text: state }), el("span", { class: "grow" }), status.running && !status.stopped && stop),
          el("p", { class: "loop-row", text: status.complete ? "The plan is complete." : "Works " + status.row + ": " + status.row_title }),
          status.stopped ? el("p", { class: "note loop-stop", text: "Stopped (" + status.stop_reason + "). Resume from the checkout with plan -resume-stop." }) : null,
          landing.commit ? el("p", { class: "note loop-landing", text: "Last landing: " + landing.subject + " (" + landing.row + ", " + landing.commit.slice(0, 12) + ", " + new Date(landing.at).toLocaleString() + ")" }) : null,
          failed ? overgo.failure(failed) : null].filter(Boolean));
      }
      // The server reads the plan and the history to answer, so one read runs at a time (an event while it
      // runs adds none), and a read that began before the loop was stopped here never draws over the stop.
      let loopReading = false, loopView = 0;
      async function readLoop() {
        if (loopReading) return;
        loopReading = true;
        const view = ++loopView;
        try {
          const status = await overgo.api.get("/loop/status", { signal: overgo.signal });
          if (view === loopView) renderLoop(status);
        } catch (err) {
          if (view === loopView && !overgo.signal.aborted && err.status !== 404) { loop.hidden = false; loop.replaceChildren(overgo.failure(err)); }
        } finally { loopReading = false; }
      }
      readLoop();

      // The served model reads by its name and how it runs; its identities wait in the line's title.
      let modelName = "";
      await overgo.read(error, () => overgo.modelInfo(), (info) => { modelName = info.model.name || ""; identity.textContent = modelName || info.model.id || ""; }, { loading: null })();

      function render(data) {
        error.replaceChildren();
        const session = data.session;
        const authority = data.authority || null;
        if (authority) {
          identity.textContent = [modelName || fmt.shortID(authority.model), authority.runtime, authority.residency].filter(Boolean).join(" · ");
          identity.title = authority.model + "\n" + authority.recipe;
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
          overgo.stat("Recipe stages", authority ? authority.stages : "—"),
          overgo.stat("Evidence", authority ? (authority.evidence || []).length : "—"),
          overgo.stat("Device", session.device));
        const table = el("table", { class: "grid" }, overgo.headerRow(["slot", "task", "state", "context", "prompt", "cached", "completion", "prefill", "decode", "per forward", "elapsed"]));
        for (const slot of data.slots) {
          const timing = slot.timings || null;
          const elapsed = timing ? Number(timing.prompt_ms) + Number(timing.predicted_ms) : null;
          table.appendChild(overgo.tableRow([slot.id, present(slot.id_task),
            el("span", { class: "tag " + (slot.is_processing ? "ok" : ""), text: slot.is_processing ? "running" : "idle" }),
            present(slot.n_ctx, fmt.grouped), present(slot.n_prompt_tokens_processed, fmt.grouped), present(slot.n_prompt_tokens_cache, fmt.grouped),
            present(timing && timing.predicted_n, fmt.grouped), present(timing && timing.prompt_per_second, (v) => Number(v).toFixed(2) + " tok/s"),
            present(timing && timing.predicted_per_second, (v) => Number(v).toFixed(2) + " tok/s"), present(timing && timing.tokens_per_forward, (v) => Number(v).toFixed(2)),
            present(elapsed, fmt.elapsed)]));
        }
        slots.replaceChildren(table);
        const recent = overgo.turnTrend(data.recent);
        trend.replaceChildren(...(recent ? [el("div", { class: "section-title", text: "Recent turns" }), recent] : []));
        summary.append(
          overgo.stat("Requested residency", authority ? authority.residency : "unknown"),
          overgo.stat("Actual residency", authority ? (authority.realized_residency || "unknown") : "unknown"));
      }

      // The sessions view listens only while it shows.
      overgo.subscribe((name, value) => { if (name === "runtime.sessions") { render(value); readLoop(); } if (name === "stream.error") error.replaceChildren(overgo.failure(value)); }, { whileShown: true });
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
          const stateClass = node.state === "failed" ? "tag caution"
            : node.state === "completed" ? "tag ok" : "tag caution";
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
          operations.replaceChildren(el("p", { class: "note empty-table", text: "Nothing is running now. Work you start (a generation, a training run, a validation) shows here while it runs, with its decisions; finished work is listed below." }), dagHost);
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
          table.appendChild(overgo.tableRow([el("span", { class: "tag " + (item.state === "completed" ? "ok" : "caution"), text: item.state }),
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
          el("span", { class: "tag " + (item.outcome === "succeeded" ? "ok" : "caution"), text: item.outcome }),
          el("span", { title: item.model, text: names.get(item.model) || fmt.shortID(item.model) }), fmt.shortID(item.recipe), measure(fmt.elapsed(Number(item.measured_ns) / 1e6)),
          fmt.grouped(item.usage.input_tokens || 0), fmt.grouped(item.usage.output_tokens || 0),
          measure(fmt.bytes(item.resources.host_to_device_bytes || 0)), measure(fmt.bytes(item.resources.device_to_host_bytes || 0))])));

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

      // A model reads by its catalogued name; one the catalog no longer lists keeps its short identity.
      const catalog = await overgo.api.get("/catalog/models", { signal: overgo.signal }).catch(() => ({ models: [] }));
      const names = overgo.modelNames(catalog.models || []);

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
