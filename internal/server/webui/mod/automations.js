(function () {
  "use strict";

  window.overgo.registerTab({
    id: "automations",
    async mount(panel, overgo) {
      const { api, el, fmt } = overgo;
      const inventoryHost = el("div");
      const operationsHost = el("div");
      const historyHost = el("div");
      const editorHost = el("div");
      const status = el("div", { class: "note" });
      panel.append(
        el("div", { class: "section-title", text: "Automations" }), status,
        el("div", { class: "section-title", text: "Inventory" }), inventoryHost,
        el("div", { class: "section-title", text: "Definition editor" }), editorHost,
        el("div", { class: "section-title", text: "Operations and approvals" }), operationsHost,
        el("div", { class: "section-title", text: "Run history" }), historyHost);

      const [definitionSchema, triggerSchema, deliverySchema] = await Promise.all([
        api.get("/workspace/schema?id=automation-definition"),
        api.get("/workspace/schema?id=automation-trigger"),
        api.get("/workspace/schema?id=automation-delivery"),
      ]);
      const definitionForm = overgo.schemaForm(definitionSchema, {});
      const triggerForm = overgo.schemaForm(triggerSchema, { kind: "manual" });
      const deliveryForm = overgo.schemaForm(deliverySchema, { kind: "artifact" });
      const publish = el("button", { class: "btn", text: "Publish definition" });
      editorHost.append(definitionForm.element, triggerForm.element, deliveryForm.element, publish);

      let inventory = [];
      function renderInventory() {
        const table = el("table", { class: "grid" }, overgo.headerRow(["name", "definition", "state", "actions"]));
        for (const item of inventory) {
          const input = el("textarea", { "aria-label": "Trigger inputs (JSON)", class: "text", rows: "2", placeholder: '{"prompt":"hello"}' });
          const destination = el("input", { "aria-label": "Allowlisted destination", class: "text", placeholder: "allowlisted destination" });
          const key = el("input", { "aria-label": "Idempotency key", class: "text", value: "manual", placeholder: "idempotency key" });
          const run = el("button", { class: "btn", text: "Run", disabled: !!item.refusal });
          const schedule = el("button", { class: "btn alt", text: "Scan schedule", disabled: !!item.refusal });
          const execute = (path) => overgo.act(status, async () => {
            const result = await api.post(path, {
              name: item.name, key: key.value, destination: destination.value,
              inputs: JSON.parse(input.value || "{}"),
            });
            // A scheduled scan answers {execution, fired}; a scan with no due slot starts nothing.
            if (result.fired === false) status.textContent = "not due / nothing started";
            else status.textContent = "accepted / " + fmt.shortID(result.operation || result.execution.operation);
          });
          run.addEventListener("click", () => execute("/automations/run"));
          schedule.addEventListener("click", () => execute("/automations/schedule"));
          table.appendChild(overgo.tableRow([item.name, fmt.shortID(item.definition), el("span", { text: item.refusal || "ready" }), el("span", {}, input, destination, key, run, schedule)]));
        }
        inventoryHost.replaceChildren(table);
      }

      const refreshInventory = overgo.read(inventoryHost, (signal) => api.get("/automations", { signal }), (value) => { inventory = value; renderInventory(); },
        { loading: "Loading automations…" });

      publish.addEventListener("click", () => overgo.act(status, async () => {
        if (!definitionForm.validate() || !triggerForm.validate() || !deliveryForm.validate()) { status.textContent = "Complete every applicable field"; return; }
        const definition = definitionForm.value();
        const created = await api.post("/automations/definitions", { name: definition.name, recipe: definition.recipe, trigger: triggerForm.value(), delivery: deliveryForm.value(), });
        await api.post("/automations/activate", { definition: created.id });
        definitionForm.markSaved(); triggerForm.markSaved(); deliveryForm.markSaved();
        status.textContent = "activated / " + fmt.shortID(created.id);
        await refreshInventory();
      }));

      function renderOperations(operations) {
        const rows = [];
        for (const item of operations || []) {
          const actions = [];
          if (!["completed", "cancelled", "failed", "blocked"].includes(item.state)) {
            actions.push(el("button", { class: "btn alt", text: "Cancel", onclick: () =>
              overgo.cancelOperation(item.id) }));
          }
          for (const recovery of (item.recovery && item.recovery.actions) || []) {
            // The shell's decision binds the advertised approval request; a refusal shows in the status line.
            const decide = (answer) => overgo.decideOperation(item.id, recovery.code, answer).catch((err) => status.replaceChildren(overgo.failure(err)));
            actions.push(el("button", { class: "btn", text: "Grant " + recovery.summary, onclick: () => decide("grant") }));
            actions.push(el("button", { class: "btn alt", text: "Decline", onclick: () => decide("decline") }));
          }
          const outputs = (item.outputs || []).map((id) => el("a", {
            class: "mono", href: overgo.contentURL(id), text: fmt.shortID(id),
          }));
          rows.push(el("div", { class: "card" },
            el("span", { class: "mono", text: fmt.shortID(item.id) }), " / " + item.state, outputs, actions));
        }
        operationsHost.replaceChildren(...rows);
      }

      const refreshHistory = overgo.read(historyHost, (signal) => api.get("/automations/history", { signal }), (history) => {
        historyHost.replaceChildren(...history.map((run) => el("div", { class: "card" },
          el("span", { class: "mono", text: fmt.shortID(run.id) }),
          " / " + run.outcome,
          ...((run.outputs || []).map((id) => el("a", {
            href: overgo.contentURL(id), text: fmt.shortID(id),
          }))))));
      }, { loading: "Loading run history…", empty: (history) => !history.length && "No automation has run yet." });

      await Promise.all([refreshInventory(), refreshHistory()]);
      // The runtime stream opens with every operation, then one event per transition, each updating its own row.
      const operations = new Map();
      overgo.subscribe((event, data) => {
        if (event === "operation.snapshot") { operations.clear(); for (const item of data || []) operations.set(item.id, item); }
        else if (event === "operation" && data.status && data.status.id) operations.set(data.status.id, data.status);
        else return;
        renderOperations([...operations.values()]);
      });
    },
  });
})();
