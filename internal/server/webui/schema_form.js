(function () {
  "use strict";

  function schemaForm(schema, initial) {
    const overgo = window.overgo;
    const { el } = overgo;
    const root = el("form", { class: "schema-form" });
    const controls = new Map();
    let baseline = JSON.stringify(initial || {});

    function readField(field, input) {
      if (input.value === "") return undefined;
      if (field.type === "string-list") return input.value.split("\n").map((item) => item.trim()).filter(Boolean);
      return input.value;
    }

    function value() {
      const result = {};
      for (const [name, control] of controls) { const current = readField(control.field, control.input); if (current !== undefined) result[name] = current; }
      return result;
    }

    function buildInput(field) {
      const input = field.type === "string-list"
        ? el("textarea", { class: "text", "aria-label": field.label, rows: "4", placeholder: "One value per line" })
        : el("input", {
          class: "text", "aria-label": field.label, type: "text",
          pattern: field.pattern || null, "data-identity-kind": field.identity_kind || null,
        });
      input.required = !!field.required;
      return input;
    }

    for (const field of schema.fields || []) {
      const input = buildInput(field);
      const supplied = initial && initial[field.name];
      if (Array.isArray(supplied)) input.value = supplied.join("\n");
      else if (supplied !== undefined && supplied !== null) input.value = String(supplied);
      const host = el("label", { class: "control" }, el("span", { text: field.label }), input);
      controls.set(field.name, { field, input });
      root.appendChild(host);
    }

    function validate() {
      if (!root.checkValidity()) return false;
      for (const control of controls.values()) {
        if (control.field.type === "string-list" && control.field.minimum_items &&
            (readField(control.field, control.input) || []).length < control.field.minimum_items) return false;
      }
      return true;
    }

    function dirty() { return JSON.stringify(value()) !== baseline; }
    function markSaved() { baseline = JSON.stringify(value()); }
    function beforeUnload(event) { if (!dirty()) return; event.preventDefault(); event.returnValue = ""; }
    window.addEventListener("beforeunload", beforeUnload);

    return {
      element: root, value, validate, dirty, markSaved,
      dispose() { window.removeEventListener("beforeunload", beforeUnload); },
    };
  }

  window.overgo.schemaForm = schemaForm;
})();
