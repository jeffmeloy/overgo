(function () {
  "use strict";
  const subscribers = new Set();
  const latest = new Map();
  let streamController = null;

  function terminal(state) { return state === "completed" || state === "cancelled" || state === "failed"; }

  function publish(name, value) {
    if (name === "operation") {
      const operations = new Map((latest.get("operation.snapshot") || []).map((item) => [item.id, item]));
      // A transition queued before a snapshot can arrive after it; a finished operation stays finished.
      const known = operations.get(value.status.id);
      if (known && terminal(known.state) && !terminal(value.status.state)) return;
      operations.set(value.status.id, value.status);
      latest.set("operation.snapshot", [...operations.values()]);
      const activity = latest.get("runtime.activity");
      if (activity) latest.set("runtime.activity", { ...activity, operations: [...operations.values()] });
    } else if (name === "runtime.serving") {
      const snapshot = latest.get("runtime.activity");
      if (!snapshot || BigInt(value.cursor) <= BigInt(snapshot.cursor)) return;
      const activity = [value.activity, ...snapshot.activity.filter((item) => item.id !== value.activity.id)];
      value = {
        ...snapshot,
        cursor: value.cursor,
        publish_failures: value.publish_failures,
        activity: activity.slice(0, snapshot.limit),
        count: Math.min(activity.length, snapshot.limit),
        truncated: snapshot.truncated || activity.length > snapshot.limit,
      };
      name = "runtime.activity";
      latest.set(name, value);
    } else if (name !== "workspace.changed") { // a change happened once; replaying it would refresh again
      latest.set(name, value);
      if (name === "operation.snapshot") {
        const activity = latest.get("runtime.activity");
        if (activity) latest.set("runtime.activity", { ...activity, operations: value });
      }
    }
    for (const subscriber of subscribers) subscriber(name, value);
  }

  async function connect() {
    const controller = new AbortController();
    streamController = controller;
    let connected = false;
    try {
      await window.overgo.api.events("/runtime/activity/stream", (name, value) => {
        if (streamController !== controller || controller.signal.aborted) return;
        if (!connected) { connected = true; latest.delete('stream.error'); publish('stream.ready', null); }
        publish(name, value);
      }, { signal: controller.signal });
      if (streamController === controller && !controller.signal.aborted) throw new Error('Activity connection closed. Retrying…');
    } catch (err) {
      if (streamController === controller && !controller.signal.aborted && err.name !== "AbortError") {
        // The idle proxy deliberately has no operation service until a model
        // is selected. Its declared refusal is not a broken connection.
        if (err.type === 'no_model_serves') { latest.delete('stream.error'); publish('stream.idle', err); }
        else publish("stream.error", err);
      }
    } finally {
      // A broken stream (a model swap replaces the serving child) reconnects
      // while anyone still listens; the pause keeps a dead server quiet.
      if (streamController === controller) {
        streamController = null;
        if (subscribers.size) setTimeout(() => { if (!streamController && subscribers.size) connect(); }, streamReconnectDelayMS);
      }
    }
  }
  const streamReconnectDelayMS = 2000;

  function subscribe(handler) {
    subscribers.add(handler);
    queueMicrotask(() => { if (subscribers.has(handler)) for (const [name, value] of latest) handler(name, value); });
    if (!streamController) connect();
    return function unsubscribe() {
      subscribers.delete(handler);
      if (!subscribers.size && streamController) { streamController.abort(); streamController = null; }
    };
  }

  function restart() { if (streamController) streamController.abort(); streamController = null; if (subscribers.size) connect(); }

  window.overgo.runtimeEvents = { subscribe, restart };
  // controlInputs renders a capability's declared controls into host: one
  // input per control, typed by the declaration; the returned map binds each
  // control name to its declaration and input. The generation tabs and the
  // front page's generation modes render from the same declaration.
  window.overgo.controlInputs = function (host, controls) {
    const el = window.overgo.el;
    const fields = new Map();
    host.replaceChildren();
    for (const control of controls || []) {
      let input;
      // A field the model's artifact exports values for (a voice) is a
      // choice, never a free input; a boolean is one too.
      const choices = (control.choices || []).length ? control.choices : (control.type === "boolean" ? ["true", "false"] : null);
      if (choices) {
        input = el("select", { class: "text", "aria-label": control.label || control.name.replace(/_/g, " ") },
          el("option", { value: "", text: control.required ? "select" : "unset", disabled: control.required, selected: true }),
          ...choices.map((choice) => el("option", { value: choice, text: choice })));
      } else {
        // A declared bound prefills the default and steps the field by the model's stride.
        const bounds = control.bounds || {};
        input = el(control.type === "text" ? "textarea" : "input", {
          class: "text", type: control.type === "integer" || control.type === "number" ? "number" : null,
          step: bounds.step || (control.type === "integer" ? "1" : "any"), required: control.required, value: control.bounds ? String(bounds.default) : null,
        });
      }
      fields.set(control.name, { control, input });
      const slot = control.type === "artifact" ? intakeStrip(control, input) : null;
      host.appendChild(el("label", { class: "control" }, el("span", { text: control.label || control.name.replace(/_/g, " ") }), input, slot));
    }
    const chips = presetChips(fields);
    if (chips) host.appendChild(chips);
    return fields;
  };

  // Presets derived from declared bounds alone: aspect ratios keep the default's pixel area over the
  // width and height strides; durations take whole seconds at the frame count's declared rate.
  const aspectPresets = [["1:1", 1, 1], ["4:3", 4, 3], ["3:2", 3, 2], ["16:9", 16, 9], ["9:16", 9, 16]];
  const durationPresets = [1, 2, 3, 5];
  function presetChips(fields) {
    const el = window.overgo.el;
    const bounded = (name) => { const field = fields.get(name); return field && field.control.bounds && field.control.bounds.step ? field : null; };
    const snap = (field, value) => { const { default: base, step } = field.control.bounds; return base + Math.round((value - base) / step) * step; };
    const chip = (text, apply) => el("button", { class: "chip", type: "button", text, onclick: apply });
    const width = bounded("width"), height = bounded("height"), frames = bounded("frames");
    const chips = [];
    if (width && height) {
      const area = width.control.bounds.default * height.control.bounds.default;
      chips.push(...aspectPresets.map(([text, horizontal, vertical]) => chip(text, () => {
        const w = snap(width, Math.sqrt(area * horizontal / vertical)); width.input.value = w; height.input.value = snap(height, area / w);
      })));
    }
    if (frames && frames.control.bounds.rate) chips.push(...durationPresets.map((seconds) => chip(seconds + " s", () => { frames.input.value = snap(frames, seconds * frames.control.bounds.rate); })));
    return chips.length ? el("div", { class: "preset-chips", role: "group", "aria-label": "Presets" }, ...chips) : null;
  }

  // intakeStrip: the store's recent stored files of the slot's media kind (attachments the
  // composer stored, media that came out of a capability), one click filling the slot.
  function intakeStrip(control, input) {
    const el = window.overgo.el;
    const strip = el("div", { class: "intake-strip", role: "group", "aria-label": "Recent " + (control.media || "stored files") });
    const kind = control.media ? control.media + "/" : "";
    const load = () => window.overgo.api.get("/artifacts?kind=file&newest=1&media=" + encodeURIComponent(kind) + "&limit=" + window.overgo.intakeStripLimit).then((listed) => {
      strip.replaceChildren(...(listed.artifacts || []).filter((item) => item.payload).map((item) => {
        const id = item.descriptor.id, source = window.overgo.contentURL(id);
        const preview = item.descriptor.media_type.startsWith("image/") ? el("img", { src: source, alt: "" }) : el("span", { class: "mono", text: item.descriptor.media_type });
        return el("button", { class: "intake-thumb", type: "button", "data-id": id, title: id, "aria-label": "use " + window.overgo.fmt.shortID(id), onclick: () => { input.value = id; input.dispatchEvent(new Event("change", { bubbles: true })); } }, preview);
      }));
    }).catch((err) => strip.replaceChildren(el("span", { class: "note", text: window.overgo.friendlyError(err) })));
    input.addEventListener("intake", load); // a fresh attachment stored into the slot relists the strip
    load();
    return strip;
  }
  // intakeStripLimit bounds the stored files a slot lists (the newest first).
  window.overgo.intakeStripLimit = 12;

  // controlValues reads the typed inputs back as the request's fields and
  // names the required ones left empty.
  window.overgo.controlValues = function (fields) {
    const input = {};
    const missing = new Set();
    for (const [name, field] of fields) {
      const value = field.input.value;
      if (value === "") { if (field.control.required) missing.add(name); continue; }
      input[name] = field.control.type === "boolean" ? value === "true" :
        (field.control.type === "integer" || field.control.type === "number" ? Number(value) : value);
    }
    return { input, missing };
  };

  // liveOperation: the first operation the stream's snapshot holds that
  // matches, or null when it holds none or the stream is down.
  function liveOperation(match) {
    return new Promise((resolve) => {
      let settled = false;
      const unsubscribe = subscribe((name, value) => {
        if (settled || !["operation.snapshot", "stream.idle", "stream.error"].includes(name)) return;
        settled = true;
        resolve(name === "operation.snapshot" ? value.find(match) || null : null);
        queueMicrotask(unsubscribe);
      });
    });
  }

  // waitOperation follows one operation on the runtime stream to its end: the
  // server's hub never drops a transition (a stream that falls behind takes
  // fresh snapshots), and a reconnect's snapshot that no longer holds the
  // operation means the server retired it.
  window.overgo.waitOperation = function (id, observe, signal) {
    return new Promise((resolve, reject) => {
      let unsubscribe = function () {}, finished = false, live = false, reconnected = false;
      function cleanup() { finished = true; unsubscribe(); if (signal) signal.removeEventListener('abort', abort); }
      function abort() { if (finished) return; cleanup(); reject(new DOMException('aborted', 'AbortError')); }
      function finish(current) { if (finished) return; if (observe) observe(current); if (!terminal(current.state)) return; cleanup(); resolve(current); }
      if (signal && signal.aborted) { abort(); return; }
      unsubscribe = subscribe((name, value) => {
        if (finished) return;
        if (name === "stream.ready" && live) reconnected = true;
        if (name === "operation" && value.status.id === id) finish(value.status);
        if (name === "operation.snapshot") {
          const current = value.find((item) => item.id === id);
          if (current) finish(current);
          else if (reconnected) { cleanup(); reject(Object.assign(new Error("operation not found"), { status: 404 })); }
        }
      });
      // The replay of the latest state runs first; a connection opened after it is a reconnect.
      queueMicrotask(() => { live = true; });
      if (signal) signal.addEventListener('abort', abort, { once: true });
    });
  };
  window.overgo.workflowWorkspace = function (definition) {
    window.overgo.registerTab({
      id: definition.id,
      async mount(panel, overgo) {
        const { api, el, clear, fmt } = overgo;
        clear(panel);
        const capabilitySelect = el("select", { class: "text", "aria-label": "capability" });
        const controls = el("div", { class: "control-grid" });
        const status = el("div", { class: "note" });
        const progress = el("progress", { class: "workflow-progress", value: 0, max: 1, hidden: true });
        const remaining = el("span", { class: "note", role: "status", "aria-label": "time remaining" });
        const trends = el("div");
        const metrics = el("div");
        const evidence = el("div");
        const run = el("button", { class: "btn", text: "Run" });
        const cancel = el("button", { class: "btn alt", text: "Cancel", hidden: true });
        // The task first, then its inputs, then the action that runs them.
        panel.append(
          el("div", { class: "section-title", text: overgo.title }),
          el("label", { class: "control" }, el("span", { text: "Task and model" }), capabilitySelect), controls,
          el("div", { class: "row" }, run, cancel), status,
          el("div", { class: "row" }, progress, remaining), trends, metrics, evidence);

        let capabilities;
        try { capabilities = await api.get("/" + definition.scope + "/capabilities"); }
        catch (err) { status.replaceChildren(overgo.failure(err)); run.disabled = true; return; }
        // A task is served by every model activated for it: the recipe is the
        // identity, the model's name the label, and a refused one says why.
        for (const capability of capabilities) {
          capabilitySelect.appendChild(el("option", {
            value: capability.recipe, disabled: !!capability.refusal, title: capability.refusal || "",
            text: capability.task + " / " + (capability.name || fmt.shortID(capability.recipe)) + (capability.refusal ? " (" + capability.refusal + ")" : ""),
          }));
        }
        let fields = new Map();
        function selected() { return capabilities.find((item) => item.recipe === capabilitySelect.value); }
        function renderControls() { const capability = selected(); fields = overgo.controlInputs(controls, capability ? capability.controls : []); }
        capabilitySelect.addEventListener("change", renderControls);
        renderControls();

        let operation = null;
        cancel.addEventListener("click", async () => {
          if (operation) {
            cancel.disabled = true;
            await overgo.cancelOperation(operation);
          }
        });
        function renderOperation(current) {
          const suffix = current.run ? " / " + fmt.shortID(current.run) : "";
          status.textContent = current.state + suffix + (current.failure ? " / " + current.failure : "");
          const total = current.progress && current.progress.total;
          const completed = current.progress ? current.progress.completed : 0;
          progress.hidden = !total;
          if (total) { progress.max = total; progress.value = completed; }
          // The remainder projects the server's elapsed wall over the steps left, as of the last step.
          const elapsed = current.progress && current.progress.elapsed_ms;
          remaining.textContent = total && completed && elapsed && !terminal(current.state) ?
            "step " + completed + " of " + total + " · about " + fmt.duration((total - completed) * elapsed / completed) + " remaining" : "";
          const drawn = (current.series || []).filter((series) => definition.trend && definition.trend(series.name) && series.values.length > 1);
          trends.replaceChildren(...drawn.map((series) => el("section", { class: "evidence-block", "data-trend": series.name },
            el("div", { class: "section-title", text: series.name + " · " + series.reports + " steps" }),
            overgo.viz.signedSeries([{ label: series.name, values: series.values, color: "var(--accent)" }]))));
          const table = overgo.table(["measurement", "value"], (current.metrics || []).map((metric) => [metric.name, String(metric.value) + (metric.unit ? " " + metric.unit : "")]), "metric-grid");
          metrics.replaceChildren(...((current.metrics || []).length ? [table] : []));
        }
        // follow: the operation's live state until it ends, then its evidence.
        async function follow(id) {
          operation = id;
          run.disabled = true;
          cancel.disabled = false;
          cancel.hidden = false;
          status.textContent = "running / " + fmt.shortID(operation);
          try {
            const completed = await overgo.waitOperation(operation, renderOperation);
            if (completed && completed.state === "completed" && definition.renderEvidence) await definition.renderEvidence(evidence, completed, overgo);
          } catch (err) { status.replaceChildren(overgo.failure(err)); } finally { operation = null; run.disabled = false; cancel.disabled = false; cancel.hidden = true; }
        }
        run.addEventListener("click", async () => {
          const capability = selected();
          if (!capability) return;
          const { input, missing } = overgo.controlValues(fields);
          if (missing.size) { const [name] = missing; fields.get(name).input.focus(); status.textContent = name + " is required"; return; }
          run.disabled = true;
          evidence.replaceChildren();
          let accepted;
          try {
            accepted = await api.post("/" + definition.scope + "/run", { task: capability.task, recipe: capability.recipe, input });
          } catch (err) { status.replaceChildren(overgo.failure(err)); run.disabled = false; return; }
          await follow(accepted.operation);
        });
        // A run of this tab's recipes still going (the page was reloaded) is followed again.
        liveOperation((item) => !terminal(item.state) && capabilities.some((capability) => capability.recipe === item.recipe)).then((live) => {
          if (!live || operation || overgo.signal.aborted) return;
          capabilitySelect.value = live.recipe;
          renderControls();
          renderOperation(live);
          follow(live.id);
        });
      },
    });
  };
})();
