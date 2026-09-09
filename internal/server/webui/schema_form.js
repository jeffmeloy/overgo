(function () {
  "use strict";

  function schemaForm(schema, initial) {
    const overgo = window.overgo;
    const { el } = overgo;
    const root = el("form", { class: "schema-form" });
    const controls = new Map();
    let baseline = JSON.stringify(initial || {});

    function applicable(field) { if (!field.when) return true; const controller = controls.get(field.when.field); return !!controller && controller.input.value === field.when.equals; }

    function readField(field, input) {
      if (!applicable(field) || input.value === "") return undefined;
      if (field.type === "boolean") return input.value === "true";
      if (field.type === "integer" || field.type === "number") return Number(input.value);
      if (field.type === "string-list") return input.value.split("\n").map((item) => item.trim()).filter(Boolean);
      return input.value;
    }

    function value() {
      const result = {};
      for (const [name, control] of controls) { const current = readField(control.field, control.input); if (current !== undefined) result[name] = current; }
      return result;
    }

    function refreshApplicability() {
      for (const control of controls.values()) {
        const active = applicable(control.field);
        control.host.hidden = !active;
        control.input.disabled = !active;
        control.input.required = active && !!control.field.required;
      }
    }

    function buildInput(field) {
      let input;
      if (field.type === "enum" || field.type === "boolean") {
        input = el("select", { class: "text", "aria-label": field.label }, el("option", { value: "", text: "Select" }));
        const options = field.type === "boolean" ?
          [{ value: "true", label: "True" }, { value: "false", label: "False" }] : field.options;
        for (const option of options || []) input.appendChild(el("option", { value: option.value, text: option.label }));
      } else if (field.type === "string-list") {
        input = el("textarea", { class: "text", rows: "4", placeholder: "One value per line" });
      } else {
        input = el("input", {
          class: "text", "aria-label": field.label, type: field.type === "integer" || field.type === "number" ? "number" : "text",
          step: field.type === "integer" ? "1" : (field.type === "number" ? "any" : null),
          pattern: field.pattern || null, min: field.minimum, max: field.maximum,
          "data-identity-kind": field.identity_kind || null,
        });
      }
      input.addEventListener("input", refreshApplicability);
      input.addEventListener("change", refreshApplicability);
      return input;
    }

    for (const field of schema.fields || []) {
      const input = buildInput(field);
      const supplied = initial && initial[field.name];
      if (Array.isArray(supplied)) input.value = supplied.join("\n");
      else if (supplied !== undefined && supplied !== null) input.value = String(supplied);
      const host = el("label", { class: "control" },
        el("span", { text: field.label + (field.unit ? " (" + field.unit + ")" : "") }), input);
      controls.set(field.name, { field, input, host });
      root.appendChild(host);
    }
    refreshApplicability();

    function validate() {
      refreshApplicability();
      if (!root.checkValidity()) return false;
      for (const control of controls.values()) {
        if (control.field.type === "string-list" && control.field.minimum_items && applicable(control.field) &&
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
