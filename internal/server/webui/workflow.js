(function () {
  "use strict";
  const subscribers = new Set();
  const latest = new Map();
  let streamController = null;

  function terminal(state) { return state === "completed" || state === "cancelled" || state === "failed"; }

  function publish(name, value) {
    if (name === "operation") {
      const operations = new Map((latest.get("operation.snapshot") || []).map((item) => [item.id, item]));
      operations.set(value.status.id, value.status);
      latest.set("operation.snapshot", [...operations.values()]);
    } else latest.set(name, value);
    for (const subscriber of subscribers) subscriber(name, value);
  }

  async function connect() {
    const controller = new AbortController();
    streamController = controller;
    try {
      await window.overgo.api.events("/runtime/activity/stream", publish, { signal: controller.signal });
    } catch (err) { if (err.name !== "AbortError") publish("stream.error", err); } finally {
      if (streamController === controller) streamController = null;
      // A broken stream (a model swap replaces the serving child) reconnects
      // while anyone still listens; the pause keeps a dead server quiet.
      if (!streamController && subscribers.size) setTimeout(() => { if (!streamController && subscribers.size) connect(); }, streamReconnectDelayMS);
    }
  }
  const streamReconnectDelayMS = 2000;

  function subscribe(handler) {
    subscribers.add(handler);
    queueMicrotask(() => {
      if (subscribers.has(handler)) for (const [name, value] of latest) handler(name, value);
    });
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
        input = el("select", { class: "text", "aria-label": control.label || control.name },
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
      host.appendChild(el("label", { class: "control" }, el("span", { text: control.label || control.name }), input, slot));
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
    return chips.length ? el("div", { class: "preset-chips", "aria-label": "presets" }, ...chips) : null;
  }

  // intakeStrip: the store's recent stored files of the slot's media kind (attachments the
  // composer stored, media that came out of a capability), one click filling the slot.
  function intakeStrip(control, input) {
    const el = window.overgo.el;
    const strip = el("div", { class: "intake-strip", "aria-label": "recent " + (control.media || "stored files") });
    const kind = control.media ? control.media + "/" : "";
    const load = () => window.overgo.api.get("/artifacts?kind=file&newest=1&media=" + encodeURIComponent(kind) + "&limit=" + window.overgo.intakeStripLimit).then((listed) => {
      strip.replaceChildren(...(listed.artifacts || []).filter((item) => item.payload).map((item) => {
        const id = item.descriptor.id, source = "/artifacts/content?id=" + encodeURIComponent(id);
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

  window.overgo.waitOperation = function (id, observe, signal) {
    return new Promise((resolve, reject) => {
      let unsubscribe = function () {};
      function finish(current) {
        if (observe) observe(current);
        if (!terminal(current.state)) return;
        unsubscribe();
        resolve(current);
      }
      unsubscribe = subscribe((name, value) => {
        if (name === "operation" && value.status.id === id) finish(value.status);
        if (name === "operation.snapshot") { const current = value.find((item) => item.id === id); if (current) finish(current); }
        if (name === "stream.error") {
          // The event stream can break under a model swap; the operation's
          // durable status still answers from the server's own wait route.
          unsubscribe();
          window.overgo.api.get("/operations/wait?id=" + encodeURIComponent(id)).then(resolve, () => reject(value));
        }
      });
      if (signal) signal.addEventListener("abort", () => {
        unsubscribe();
        reject(new DOMException("aborted", "AbortError"));
      }, { once: true });
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
        const progress = el("progress", { class: "workflow-progress", value: 0, max: 1, style: "display:none" });
        const metrics = el("div");
        const evidence = el("div");
        const run = el("button", { class: "btn", text: "Run" });
        const cancel = el("button", { class: "btn alt", text: "Cancel", style: "display:none" });
        panel.append(
          el("div", { class: "section-title", text: definition.label }),
          el("div", { class: "row" }, capabilitySelect, run, cancel), controls, status,
          progress, metrics, evidence);

        let capabilities;
        try { capabilities = await api.get("/" + definition.scope + "/capabilities"); }
        catch (err) {
          status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
          run.disabled = true;
          return;
        }
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
            await api.post("/operations/cancel", { id: operation });
          }
        });
        function renderOperation(current) {
          const suffix = current.run ? " / " + fmt.shortID(current.run) : "";
          status.textContent = current.state + suffix + (current.failure ? " / " + current.failure : "");
          const total = current.progress && current.progress.total;
          progress.style.display = total ? "" : "none";
          if (total) { progress.max = total; progress.value = current.progress.completed; }
          const table = overgo.table(["measurement", "value"], (current.metrics || []).map((metric) => [metric.name, String(metric.value) + (metric.unit ? " " + metric.unit : "")]), "metric-grid");
          metrics.replaceChildren(...((current.metrics || []).length ? [table] : []));
        }
        run.addEventListener("click", async () => {
          const capability = selected();
          if (!capability) return;
          const { input, missing } = overgo.controlValues(fields);
          if (missing.size) {
            const [name] = missing;
            fields.get(name).input.focus();
            status.textContent = name + " is required";
            return;
          }
          run.disabled = true;
          cancel.disabled = false;
          cancel.style.display = "";
          evidence.replaceChildren();
          try {
            const accepted = await api.post("/" + definition.scope + "/run", {
              task: capability.task, recipe: capability.recipe, input,
            });
            operation = accepted.operation;
            status.textContent = "running / " + fmt.shortID(operation);
            const completed = await overgo.waitOperation(operation, renderOperation);
            if (completed && completed.state === "completed" && definition.renderEvidence) await definition.renderEvidence(evidence, completed, overgo);
          } catch (err) { status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); } finally {
            operation = null;
            run.disabled = false;
            cancel.disabled = false;
            cancel.style.display = "none";
          }
        });
      },
    });
  };
})();
