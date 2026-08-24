(function () {
  "use strict";
  const subscribers = new Set();
  const latest = new Map();
  let streamController = null;

  function terminal(state) {
    return state === "completed" || state === "cancelled" || state === "failed";
  }

  function publish(name, value) {
    if (name === "operation") {
      const operations = new Map((latest.get("operation.snapshot") || []).map((item) => [item.id, item]));
      operations.set(value.status.id, value.status);
      latest.set("operation.snapshot", [...operations.values()]);
    } else {
      latest.set(name, value);
    }
    for (const subscriber of subscribers) subscriber(name, value);
  }

  async function connect() {
    const controller = new AbortController();
    streamController = controller;
    try {
      await window.overgo.api.events("/runtime/activity/stream", publish, { signal: controller.signal });
    } catch (err) {
      if (err.name !== "AbortError") publish("stream.error", err);
    } finally {
      if (streamController === controller) streamController = null;
    }
  }

  function subscribe(handler) {
    subscribers.add(handler);
    queueMicrotask(() => {
      if (subscribers.has(handler)) for (const [name, value] of latest) handler(name, value);
    });
    if (!streamController) connect();
    return function unsubscribe() {
      subscribers.delete(handler);
      if (!subscribers.size && streamController) {
        streamController.abort();
        streamController = null;
      }
    };
  }

  window.overgo.runtimeEvents = { subscribe };
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
        if (name === "operation.snapshot") {
          const current = value.find((item) => item.id === id);
          if (current) finish(current);
        }
        if (name === "stream.error") {
          unsubscribe();
          reject(value);
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
      label: definition.label,
      section: definition.section,
      async mount(panel, overgo) {
        const { api, el, clear, fmt } = overgo;
        clear(panel);
        const capabilitySelect = el("select", { class: "text" });
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
        for (const capability of capabilities) {
          capabilitySelect.appendChild(el("option", {
            value: capability.task,
            text: capability.task + " / " + fmt.shortID(capability.recipe),
          }));
        }
        let fields = new Map();
        function selected() { return capabilities.find((item) => item.task === capabilitySelect.value); }
        function renderControls() {
          fields = new Map();
          controls.replaceChildren();
          const capability = selected();
          if (!capability) return;
          for (const control of capability.controls) {
            let input;
            if (control.type === "boolean") {
              input = el("select", { class: "text" },
                el("option", { value: "", text: control.required ? "select" : "unset", disabled: control.required, selected: true }),
                el("option", { value: "true", text: "true" }),
                el("option", { value: "false", text: "false" }));
            } else {
              input = el(control.type === "text" ? "textarea" : "input", {
                class: "text", type: control.type === "integer" || control.type === "number" ? "number" : null,
                step: control.type === "integer" ? "1" : "any", required: control.required,
              });
            }
            fields.set(control.name, { control, input });
            controls.appendChild(el("label", { class: "control" }, el("span", { text: control.name }), input));
          }
        }
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
          if (total) {
            progress.max = total;
            progress.value = current.progress.completed;
          }
          const table = el("table", { class: "grid metric-grid" });
          table.appendChild(el("tr", {}, el("th", { text: "measurement" }), el("th", { text: "value" })));
          for (const metric of current.metrics || []) {
            table.appendChild(el("tr", {},
              el("td", { text: metric.name }),
              el("td", { class: "mono", text: String(metric.value) + (metric.unit ? " " + metric.unit : "") })));
          }
          metrics.replaceChildren(...((current.metrics || []).length ? [table] : []));
        }
        run.addEventListener("click", async () => {
          const capability = selected();
          if (!capability) return;
          const input = {};
          for (const [name, field] of fields) {
            const value = field.input.value;
            if (value === "" && field.control.required) {
              field.input.focus();
              status.textContent = name + " is required";
              return;
            }
            if (value === "") continue;
            input[name] = field.control.type === "boolean" ? value === "true" :
              (field.control.type === "integer" || field.control.type === "number" ? Number(value) : value);
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
            if (completed && completed.state === "completed" && definition.renderEvidence) {
              await definition.renderEvidence(evidence, completed, overgo);
            }
          } catch (err) {
            status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
          } finally {
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
