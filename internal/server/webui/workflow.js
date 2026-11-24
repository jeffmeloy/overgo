(function () {
  "use strict";
  const subscribers = new Set();
  const latest = new Map();
  let streamController = null;
  let synchronized = false;

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
    } else if (name === "resync") {
      // The page's first snapshot lost nothing: it read everything it shows just now.
      const first = !synchronized;
      synchronized = true;
      if (first) return;
    } else if (name !== "workspace.changed" && name !== "agent.output") { // a change or an output piece happened once; replaying it would repeat it
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
  // A dataset control that declares a catalog format is a picker of the catalogued datasets holding it;
  // one that declares none (a preference file the catalog does not hold) takes an identity typed in.
  const picksDataset = (control) => control.type === "dataset" && !!control.format;

  // offerDataset lists a dataset's option under its identity: a name that arrives later relabels the
  // option a remembered choice already put there.
  function offerDataset(select, id, text) {
    const found = [...select.options].find((option) => option.value === id);
    if (found) found.text = text;
    else select.appendChild(window.overgo.el("option", { value: id, text }));
  }

  // datasetPicker: the registered datasets holding the control's format, by name; the value is the dataset's
  // identity. (The format is the control's own declaration: media names an artifact slot's file kind.)
  function datasetPicker(control) {
    const { api, el, fmt, friendlyError } = window.overgo;
    const placeholder = el("option", { value: "", text: "select", disabled: control.required, selected: true });
    const select = el("select", { class: "text", "aria-label": control.label || control.role || control.name.replace(/_/g, " ") }, placeholder);
    api.get("/datasets").then((catalog) => {
      const offered = (catalog.datasets || []).filter((entry) => entry.available && (entry.formats || []).includes(control.format));
      for (const entry of offered) offerDataset(select, entry.dataset, entry.name + " · " + fmt.grouped(entry.files) + " files");
      if (!offered.length && select.options.length === 1) placeholder.text = "none registered: import one in the Library";
    }).catch((err) => { placeholder.text = err.status === 404 ? "none registered: import one in the Library" : friendlyError(err); });
    return select;
  }

  // controlInputs renders a capability's declared controls into host: one
  // input per control, typed by the declaration; the returned map binds each
  // control name to its declaration and input. The generation tabs and the
  // front page's generation modes render from the same declaration.
  // An object or a list control is set part by part: an object field by field (a MOSS section's schedule), a list item
  // by item (the adapters a section stacks).
  const setPartByPart = (control) => control.type === "object" || control.type === "list";
  // controlLabel: the name a page shows a control by.
  const controlLabel = (control) => control.label || control.role || control.name.replace(/_/g, " ");

  // controlInputs renders a capability's declared controls into host: one
  // input per control, typed by the declaration; the returned map binds each
  // control name to its declaration and input. The generation tabs and the
  // front page's generation modes render from the same declaration. An
  // object's fields render through it too (nested), with no disclosure or
  // chips of their own.
  window.overgo.controlInputs = function (host, controls, nested) {
    const el = window.overgo.el;
    const fields = new Map();
    const advanced = el("details", { class: "advanced-controls" }, el("summary", { text: "Advanced options" }));
    // A group of tuning behind the disclosure (a MOSS request's sampling) is a part of it of its own, made where its
    // first control stands (moss-page/groups).
    const groups = new Map();
    const groupOf = (name) => {
      if (!groups.has(name)) {
        const grid = el("div", { class: "control-grid" });
        advanced.appendChild(el("fieldset", { class: "control-group advanced-group", "data-group": name }, el("legend", { text: name }), grid));
        groups.set(name, grid);
      }
      return groups.get(name);
    };
    host.replaceChildren();
    for (const control of controls || []) {
      const input = controlInput(control);
      fields.set(control.name, { control, input });
      const slot = artifactSlot(control, input);
      let target = host;
      if (!nested && control.advanced && !control.required) target = control.group ? groupOf(control.group) : advanced;
      target.appendChild(setPartByPart(control) ? input : el("label", { class: "control" }, el("span", { text: controlLabel(control) }), input, slot));
      if (control.help) target.appendChild(el('details', { class: 'control-help' }, el('summary', { text: 'About ' + controlLabel(control) }), el('p', { class: 'note', text: control.help })));
    }
    if (nested) return fields;
    const chips = presetChips(fields);
    if (chips) host.appendChild(chips);
    if (advanced.children.length > 1) host.appendChild(advanced);
    choiceShows(fields, host);
    presetSettings(fields);
    // A canvas of boxes takes the picture's shape, its width and height as set or as declared, as they change.
    const width = fields.get("width"), height = fields.get("height");
    const canvases = [...host.querySelectorAll(".box-canvas")];
    if (canvases.length && width && height) {
      const side = (field) => Number(field.input.value || field.input.placeholder);
      const shape = () => { for (const canvas of canvases) canvas.shape(side(width), side(height)); };
      for (const name of ["input", "change"]) host.addEventListener(name, shape);
      shape();
    }
    return fields;
  };

  // controlInput: the one input a control is set by, typed by its declaration.
  function controlInput(control) {
    const el = window.overgo.el;
    if (control.type === "object") return objectInput(control);
    if (control.type === "list") return listInput(control);
    // A field the model's artifact exports values for (a voice) is a
    // choice, never a free input; a boolean is one too.
    const choices = (control.choices || []).length ? control.choices : (control.type === "boolean" ? ["true", "false"] : null);
    if (picksDataset(control)) return datasetPicker(control);
    if (choosesPicture(control)) return pictureValue();
    if (choices) {
      return el("select", { class: "text", "aria-label": controlLabel(control) },
        // A field left unset is given the default its control declares where the request is taken in.
        el("option", { value: "", text: control.required ? "select" : control.default ? "default (" + control.default + ")" : "unset", disabled: control.required, selected: true }),
        // A choice the declaration names plainly (a MOSS Ultra section) is offered by that name, its value the choice.
        ...choices.map((choice) => el("option", { value: choice, text: (control.names && control.names[choice]) || choice })));
    }
    // A declared bound steps the field by the model's stride. Its default fills a required field and the picture's
    // geometry, which no preset chooses; in a sampling field a request may leave out it is a hint, so a request
    // carries what the person set and a preset chosen beside it is not overridden by values the person never
    // touched (SQA-340, preset-form/do).
    const bounds = control.bounds || {};
    const declared = control.bounds ? String(bounds.default) : null;
    const filled = control.required || geometryControls.has(control.name);
    return el(control.type === "text" ? "textarea" : "input", {
      class: "text", type: control.type === "integer" || control.type === "number" ? "number" : null,
      step: bounds.step || (control.type === "integer" ? "1" : "any"), required: control.required,
      value: filled ? declared : null, placeholder: filled ? null : declared,
    });
  }

  // partsValue gives a group the value a typed input has, so remembered entries and presets set and read it alike: the
  // parts set, as JSON, or empty where none is.
  function partsValue(group, read, write) {
    Object.defineProperty(group, "value", {
      get() { const value = read(); return value === null ? "" : JSON.stringify(value); },
      set(text) { let value = null; try { value = text ? JSON.parse(text) : null; } catch (_) { /* a malformed entry sets nothing */ } write(value); },
    });
    return group;
  }

  // partText is a value the request holds as its input shows it: an object or a list as JSON, a scalar as its text.
  const partText = (value) => value === undefined || value === null ? "" : (typeof value === "object" ? JSON.stringify(value) : String(value));

  // setPart sets one part's input to a value the request holds; a value a choice does not offer is left unset.
  function setPart(field, value) {
    const text = partText(value);
    field.input.value = field.control.choices?.length && text && !field.control.choices.includes(text) ? "" : text;
  }

  // choiceShows: a choice control that declares the controls each of its choices shows (a MOSS prompt mode: the prompt
  // and negative as written or enhanced, the layout for the builder) shows the chosen one's alone, its declared default
  // where none is chosen; a control hidden so is not sent (controlValues), so the request is the chosen mode's
  // (moss-prompt-mode). A MOSS section shows the fields it runs that another does not alike (moss-page/groups), and a
  // group of tuning shows while a control in it does.
  function choiceShows(fields, host) {
    for (const [, field] of fields) {
      const shows = field.control.shows;
      if (!shows) continue;
      const governed = new Set(Object.values(shows).flat());
      const apply = () => {
        const shown = shows[field.input.value];
        for (const name of governed) {
          const other = fields.get(name);
          if (!other) continue;
          other.hidden = !!shown && !shown.includes(name);
          controlElement(other).hidden = other.hidden;
        }
        if (shown) takePlaces(shown.map((name) => fields.get(name)).filter(Boolean), [...governed].map((name) => fields.get(name)).filter((other) => other && other.hidden));
        for (const group of host.querySelectorAll(".advanced-group")) {
          group.hidden = ![...group.querySelector(".control-grid").children].some((part) => !part.hidden && !part.matches(".control-help"));
        }
      };
      field.input.addEventListener("change", apply);
      field.input.addEventListener("input", apply);
      apply();
    }
  }

  // controlElement: what a control stands as in the form, its label or its group.
  const controlElement = (field) => field.input.closest("label.control") || field.input;

  // takePlaces: a field a choice shows from behind the disclosure, in place of one the choice hides from the form, takes
  // that one's place: the builder's layout where the prompt was, a depth section's control where the picture was.
  function takePlaces(shown, hidden) {
    const replaced = hidden.find((field) => !controlElement(field).closest(".advanced-controls"));
    if (!replaced) return;
    for (const field of shown) {
      if (controlElement(field).closest(".advanced-controls")) controlElement(replaced).before(controlElement(field));
    }
  }

  // presetSettings: choosing a preset sets each control it fills to its value (a MOSS Ultra section's steps, schedule,
  // denoise, adapters and size) and returns what the preset chosen before it filled, and the person left as it was, to
  // what it held before any preset (a size its declared default), so the page shows what a preset runs; the request
  // still names the preset for what no control offers.
  function presetSettings(fields) {
    const preset = fields.get("preset");
    const settings = preset && preset.control.settings;
    if (!settings) return;
    let filled = {};
    const before = new Map();
    // presetFilled: the field still holds what the last preset filled, not what a person set.
    const presetFilled = (name, field) => Object.hasOwn(filled, name) && field.input.value === partText(filled[name]);
    preset.input.addEventListener("change", () => {
      const chosen = settings[preset.input.value] || {};
      for (const [name, field] of fields) {
        if (name === "preset") continue;
        if (Object.hasOwn(chosen, name)) {
          // What a person set is kept at each choice of a preset, so its release gives back the latest (SQA-462).
          if (!presetFilled(name, field)) before.set(name, field.input.value);
          setPart(field, chosen[name]);
        } else if (presetFilled(name, field)) field.input.value = before.get(name) ?? "";
      }
      filled = chosen;
    });
  }

  // objectInput: an object control's fields, a group of controls of their own.
  function objectInput(control) {
    const el = window.overgo.el;
    const host = el("div", { class: "control-grid" });
    const group = el("fieldset", { class: "control-group", "data-control": control.name }, el("legend", { text: controlLabel(control) }), host);
    const fields = window.overgo.controlInputs(host, control.fields, true);
    return partsValue(group, () => { const { input } = window.overgo.controlValues(fields); return Object.keys(input).length ? input : null; },
      (value) => { for (const [name, field] of fields) setPart(field, value ? value[name] : null); });
  }

  // listInput: a list control's items, rows a person adds and removes, each the item's control; a list of boxes on the
  // picture is drawn and drawn on besides (boxEditor).
  function listInput(control) {
    const el = window.overgo.el;
    const item = { ...control.item, name: control.name };
    const rows = el("div", { class: "control-items" });
    const inputs = [];
    const values = () => inputs.map((input) => partValue(item, input.value)).filter((value) => value !== undefined);
    const changed = () => rows.dispatchEvent(new Event("change", { bubbles: true }));
    const add = (value) => {
      const input = controlInput(item);
      // An item's input stands in no label of its own, so it is named for its list.
      if (!setPartByPart(item)) input.setAttribute("aria-label", controlLabel(control));
      const row = el("div", { class: "control-item" }, input, el("button", { class: "btn alt", type: "button", text: "Remove", "aria-label": "Remove " + controlLabel(control), onclick: () => {
        inputs.splice(inputs.indexOf(input), 1); row.remove(); changed();
      } }));
      setPart({ control: item, input }, value);
      inputs.push(input);
      rows.appendChild(row);
    };
    const group = el("fieldset", { class: "control-group control-list", "data-control": control.name }, el("legend", { text: controlLabel(control) }), rows,
      el("button", { class: "btn alt", type: "button", text: "Add", "aria-label": "Add " + controlLabel(control), onclick: () => { add(null); changed(); } }));
    const editor = boxItem(control.item) ? boxEditor(values, (box) => { add(box); changed(); }) : null;
    if (editor) {
      group.insertBefore(editor, rows);
      for (const name of ["input", "change"]) group.addEventListener(name, editor.draw);
    }
    return partsValue(group, () => { const held = values(); return held.length ? held : null; },
      (value) => { inputs.length = 0; rows.replaceChildren(); for (const entry of Array.isArray(value) ? value : []) add(entry); if (editor) editor.draw(); });
  }

  // boxItem says a list's items are boxes on the picture, each a corner and an extent as shares of its canvas (a MOSS
  // layout's regions, as the MOSS Ultra workflow's prompt builder draws them), which a page draws on.
  const boxItem = (item) => item.type === "object" && ["x", "y", "w", "h"].every((name) => (item.fields || []).some((field) => field.name === name));

  // boxEditor: a list's boxes on a canvas of the picture's shape, each numbered as its row; dragging across the canvas
  // adds a box, its corner and extent as shares of the canvas to three places, as the builder's 0-1000 grid holds them.
  // shape(width, height) gives the canvas the picture's shape.
  function boxEditor(boxes, add) {
    const el = window.overgo.el;
    const canvas = el("div", { class: "box-canvas", role: "application", "aria-label": "Draw a region" });
    const place = (node, box) => Object.assign(node.style, { left: box.x * 100 + "%", top: box.y * 100 + "%", width: box.w * 100 + "%", height: box.h * 100 + "%" });
    canvas.draw = () => canvas.replaceChildren(...boxes().map((box, index) => {
      const drawn = el("div", { class: "box", text: String(index + 1) });
      place(drawn, box);
      return drawn;
    }));
    canvas.shape = (width, height) => { canvas.style.aspectRatio = width > 0 && height > 0 ? width + " / " + height : ""; };
    const share = (event) => {
      const bounds = canvas.getBoundingClientRect();
      return [Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1), Math.min(Math.max((event.clientY - bounds.top) / bounds.height, 0), 1)];
    };
    const round = (value) => Math.round(value * 1000) / 1000;
    const spanned = (from, to) => ({ x: round(Math.min(from[0], to[0])), y: round(Math.min(from[1], to[1])), w: round(Math.abs(to[0] - from[0])), h: round(Math.abs(to[1] - from[1])) });
    let from = null, drawing = null;
    canvas.addEventListener("pointerdown", (event) => {
      from = share(event);
      drawing = el("div", { class: "box drawing" });
      canvas.appendChild(drawing);
      if (canvas.setPointerCapture && event.isPrimary) try { canvas.setPointerCapture(event.pointerId); } catch (_) { /* a synthetic pointer holds no capture */ }
    });
    canvas.addEventListener("pointermove", (event) => { if (from) place(drawing, spanned(from, share(event))); });
    canvas.addEventListener("pointerup", (event) => {
      if (!from) return;
      const box = spanned(from, share(event));
      from = null;
      drawing.remove();
      if (box.w > 0 && box.h > 0) add(box);
    });
    canvas.draw();
    return canvas;
  }

  // partValue is a control's entry as the request holds it: nothing where it is empty, a boolean, a number, an object
  // or a list as typed, text as it is.
  function partValue(control, text) {
    if (text === "") return undefined;
    if (control.type === "boolean") return text === "true";
    if (control.type === "integer" || control.type === "number") return Number(text);
    if (setPartByPart(control)) return JSON.parse(text);
    return text;
  }

  // geometryControls are the bounded fields that size the picture or clip, which its declared default fills: a request
  // without them names no size (a layout prompt's canvas read 0 x 0, an oscillator's clip at scale 0 was refused, the
  // model-journey lane on 2026-11-17), and the chips below set the picture's and the clip's.
  const geometryControls = new Set(["width", "height", "frames", "scale"]);

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

  // choosesPicture says an artifact slot takes a picture (a MOSS picture, its depth map, a question's image), which a
  // page has chosen, never typed (moss-page/pictures).
  const choosesPicture = (control) => control.type === "artifact" && control.media === "image";

  // artifactSlot: what stands beside an artifact slot's input, a picture's chooser or the store's recent files of the
  // slot's kind; nothing beside any other control.
  function artifactSlot(control, input) {
    if (control.type !== "artifact") return null;
    return choosesPicture(control) ? pictureChooser(control, input) : intakeStrip(control, input);
  }

  // pictureValue: the input a picture slot holds its chosen picture's stored identity in, never shown or typed; any
  // setting of it, by its chooser, a preset, a remembered entry or an attachment routed to it, says so to the chooser.
  function pictureValue() {
    const input = window.overgo.el("input", { type: "hidden" });
    const native = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
    Object.defineProperty(input, "value", {
      get() { return native.get.call(input); },
      set(id) { native.set.call(input, id); input.dispatchEvent(new Event("picture")); },
    });
    return input;
  }

  // pictureChooser: a picture slot's chooser: the chosen picture's thumbnail, a picture file uploaded or dropped on it,
  // stored and named by its identity, and the store's recent pictures by their thumbnails (intakeStrip).
  function pictureChooser(control, input) {
    const { el, api, contentURL, friendlyError } = window.overgo;
    const name = controlLabel(control);
    const shown = el("div", { class: "picture-chosen" });
    const note = el("span", { class: "note", role: "status" });
    const file = el("input", { type: "file", accept: "image/*", hidden: true, "aria-label": "Upload the " + name });
    const choose = (id) => { input.value = id; input.dispatchEvent(new Event("change", { bubbles: true })); };
    const upload = el("button", { class: "btn alt", type: "button", text: "Upload", "aria-label": "Upload the " + name, onclick: () => file.click() });
    const clear = el("button", { class: "btn alt", type: "button", text: "Clear", "aria-label": "Clear the " + name, onclick: () => choose("") });
    const draw = () => {
      clear.hidden = !input.value;
      shown.replaceChildren(input.value ? el("img", { src: contentURL(input.value), alt: "The " + name + " chosen", class: "picture-thumb", title: input.value })
        : el("span", { class: "note", text: "None chosen: upload one, drop one here, or pick a recent one" }));
    };
    // store keeps a picture file and chooses it, and relists the recent pictures.
    const store = async (picked) => {
      if (!picked || !picked.type.startsWith("image/")) { note.textContent = "Upload or drop a picture file"; return; }
      note.textContent = "Storing " + picked.name + "…";
      try {
        const stored = await api.upload("/artifacts/intake", picked);
        note.textContent = "";
        choose(stored.id);
        input.dispatchEvent(new Event("intake"));
      } catch (err) { note.textContent = friendlyError(err); }
    };
    file.addEventListener("change", () => { store(file.files[0]); file.value = ""; });
    const chooser = el("div", { class: "picture-chooser" }, shown, el("div", { class: "row" }, upload, clear, note), file, intakeStrip(control, input));
    // A click on the thumbnail or around it chooses nothing: the label holding the chooser would pass it to Upload.
    chooser.addEventListener("click", (event) => { if (!event.target.closest("button, input")) event.preventDefault(); });
    chooser.addEventListener("dragover", (event) => { event.preventDefault(); chooser.classList.add("dropping"); });
    chooser.addEventListener("dragleave", () => chooser.classList.remove("dropping"));
    chooser.addEventListener("drop", (event) => { event.preventDefault(); chooser.classList.remove("dropping"); store(event.dataTransfer && event.dataTransfer.files[0]); });
    input.addEventListener("picture", draw);
    draw();
    return chooser;
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

  // rememberedControls: the last values a capability ran with and the reader's named presets, kept per
  // capability recipe in this browser; the declared defaults stay the fallback. Storage a browser refuses
  // (a private window) keeps nothing, and every read then answers nothing remembered.
  const rememberedPrefix = "overgo.controls.";
  function readRemembered(recipe) {
    try { return JSON.parse(localStorage.getItem(rememberedPrefix + recipe) || "null") || { presets: {} }; }
    catch (_) { return { presets: {} }; }
  }
  function writeRemembered(recipe, kept) {
    try { localStorage.setItem(rememberedPrefix + recipe, JSON.stringify(kept)); } catch (_) { /* storage refused */ }
  }
  window.overgo.rememberedControls = {
    last: (recipe) => readRemembered(recipe).last || null,
    remember(recipe, values) { const kept = readRemembered(recipe); kept.last = values; writeRemembered(recipe, kept); },
    presets: (recipe) => readRemembered(recipe).presets || {},
    savePreset(recipe, name, values) { const kept = readRemembered(recipe); kept.presets = { ...(kept.presets || {}), [name]: values }; writeRemembered(recipe, kept); },
    // values: the fields' settings as typed; an artifact slot holds one turn's input, not a setting.
    values: (fields) => Object.fromEntries([...fields].filter(([, field]) => field.control.type !== "artifact").map(([name, field]) => [name, field.input.value])),
    // apply: values onto the fields; a value a choice no longer offers is left unset.
    apply(fields, values) {
      if (!values) return;
      for (const [name, field] of fields) {
        if (!Object.hasOwn(values, name)) continue;
        const value = String(values[name]);
        // A dataset's name may not have arrived yet: its identity stands in until it does.
        if (picksDataset(field.control) && value) offerDataset(field.input, value, window.overgo.fmt.shortID(value));
        field.input.value = field.control.choices?.length && !field.control.choices.includes(value) ? "" : value;
      }
    },
  };

  // presetControls: the capability's named presets beside its controls: choosing one fills the fields,
  // and Save preset keeps the current entries under a name.
  window.overgo.presetControls = function (recipe, fields) {
    const el = window.overgo.el;
    const remembered = window.overgo.rememberedControls;
    const choose = el("select", { class: "text w-auto", "aria-label": "preset" });
    const name = el("input", { class: "text", type: "text", "aria-label": "Preset name", placeholder: "Preset name" });
    function list() {
      choose.replaceChildren(el("option", { value: "", text: "Presets" }), ...Object.keys(remembered.presets(recipe)).sort().map((preset) => el("option", { value: preset, text: preset })));
    }
    choose.addEventListener("change", () => { if (choose.value) remembered.apply(fields, remembered.presets(recipe)[choose.value]); });
    const save = el("button", { class: "btn alt", type: "button", text: "Save preset", onclick: () => {
      const label = name.value.trim();
      if (!label) { name.focus(); return; }
      remembered.savePreset(recipe, label, remembered.values(fields));
      name.value = "";
      list();
      choose.value = label;
    } });
    list();
    return el("span", { class: "row presets" }, choose, name, save);
  };

  // controlValues reads the typed inputs back as the request's fields, names the required ones left empty, and refuses a
  // whole number the browser cannot carry exactly: past 2^53 a JSON number arrives rounded, so a seed pasted from a run
  // would silently draw another picture.
  window.overgo.controlValues = function (fields) {
    const input = {};
    const missing = new Set();
    let refused = "", refusal = "";
    for (const [name, field] of fields) {
      // A control the chosen mode hides is not the request's (choiceShows).
      if (field.hidden) continue;
      const value = field.input.value;
      if (value === "") { if (field.control.required) missing.add(name); continue; }
      if (field.control.type === "integer" && !Number.isSafeInteger(Number(value))) {
        if (!refused) { refused = name; refusal = name + " must be a whole number from " + Number.MIN_SAFE_INTEGER + " to " + Number.MAX_SAFE_INTEGER; }
        continue;
      }
      input[name] = partValue(field.control, value);
    }
    return { input, missing, refused, refusal };
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

  // waitOperation follows one operation on the runtime stream to its end: a
  // stream that falls behind takes a fresh snapshot in place of what it missed,
  // terminal transitions included, so a snapshot that no longer holds an
  // operation means the server retired it: after a reconnect, or once this wait
  // has seen it. A snapshot from before the wait's operation existed means nothing.
  window.overgo.waitOperation = function (id, observe, signal) {
    return new Promise((resolve, reject) => {
      let unsubscribe = function () {}, finished = false, live = false, reconnected = false, seen = false;
      function cleanup() { finished = true; unsubscribe(); if (signal) signal.removeEventListener('abort', abort); }
      function abort() { if (finished) return; cleanup(); reject(new DOMException('aborted', 'AbortError')); }
      function finish(current) { if (finished) return; seen = true; if (observe) observe(current); if (!terminal(current.state)) return; cleanup(); resolve(current); }
      if (signal && signal.aborted) { abort(); return; }
      unsubscribe = subscribe((name, value) => {
        if (finished) return;
        if (name === "stream.ready" && live) reconnected = true;
        if (name === "operation" && value.status.id === id) finish(value.status);
        if (name === "operation.snapshot") {
          const current = value.find((item) => item.id === id);
          if (current) finish(current);
          else if (reconnected || seen) { cleanup(); reject(Object.assign(new Error("operation not found"), { status: 404 })); }
        }
      });
      // The replay of the latest state runs first; a connection opened after it is a reconnect.
      queueMicrotask(() => { live = true; });
      if (signal) signal.addEventListener('abort', abort, { once: true });
    });
  };
  // runNotice tells a person a run has ended while the page is not in front. A final state seen on a
  // hidden document puts the count of runs ended since before the tab's title until the page is shown
  // again, and raises a browser notification when its permission is granted. The permission is asked
  // once, from the click that starts a long run, which is the only place a browser takes the request.
  const noticeAskedKey = "overgo.notice.asked";
  let endedHidden = 0, shownTitle = "";
  document.addEventListener("visibilitychange", () => {
    if (document.hidden || !endedHidden) return;
    endedHidden = 0;
    document.title = shownTitle;
  });
  window.overgo.runNotice = {
    ask() {
      if (!("Notification" in window) || Notification.permission !== "default") return;
      try {
        if (localStorage.getItem(noticeAskedKey)) return;
        localStorage.setItem(noticeAskedKey, "1");
      } catch (_) { /* storage refused: the request is made again at the next long run */ }
      Notification.requestPermission();
    },
    ended(label, state) {
      if (!document.hidden) return;
      if (!endedHidden) shownTitle = document.title;
      endedHidden++;
      document.title = "(" + endedHidden + ") " + shownTitle;
      if (!("Notification" in window) || Notification.permission !== "granted") return;
      try {
        const notice = new Notification(label + " " + state);
        notice.addEventListener("click", () => { window.focus(); notice.close(); });
      } catch (_) { /* a browser that refuses a page's notification keeps the title mark */ }
    },
  };
  window.overgo.workflowWorkspace = function (definition) {
    window.overgo.registerTab({
      id: definition.id,
      async mount(panel, overgo) {
        const { api, el, clear, fmt } = overgo;
        clear(panel);
        const capabilitySelect = el("select", { class: "text", "aria-label": "capability" });
        const controls = el("div", { class: "control-grid" });
        // placement: where the selected capability runs, as it declares.
        const placement = el("p", { class: "note workflow-placement" });
        const status = el("div", { class: "note", role: "status", "aria-label": "run status" });
        const progress = el("progress", { class: "workflow-progress", value: 0, max: 1, hidden: true });
        const remaining = el("span", { class: "note", role: "status", "aria-label": "time remaining" });
        const trends = el("div");
        const metrics = el("div");
        const evidence = el("div");
        // previewHost: the run's input as the workflow reads it, when the tab declares a preview.
        const previewHost = el("div");
        // sizeHost: the bytes the run would write, when the tab's server sizes a submission before it runs (definition.sized).
        const sizeHost = el("div");
        const run = el("button", { class: "btn", text: "Run" });
        const cancel = el("button", { class: "btn alt", text: "Cancel", hidden: true });
        // command: the commands that reproduce this submission outside the page; a changed entry clears them.
        const command = el("button", { class: "btn alt", text: "Show command" });
        const commands = el("div", { class: "workflow-commands", role: "group", "aria-label": "Reproducing commands" });
        // The task first, then its inputs, then the action that runs them, which stays in view as the form scrolls; a run's
        // status, progress and result stand beside the form on a wide screen and below it on a narrow one
        // (moss-page/result).
        panel.append(
          el("div", { class: "section-title", text: overgo.title }),
          el("div", { class: "workflow-columns" },
            el("div", { class: "workflow-form" },
              el("div", { class: "workflow-task" }, el("label", { class: "control" }, el("span", { text: "Task and model" }), capabilitySelect), placement), controls, previewHost, sizeHost,
              el("div", { class: "row workflow-actions" }, run, cancel, command), commands),
            el("section", { class: "workflow-result", "aria-label": "Run" }, status,
              el("div", { class: "row" }, progress, remaining), trends, metrics, evidence)));

        let capabilities;
        try { capabilities = await api.get("/" + definition.scope + "/capabilities"); }
        catch (err) { status.replaceChildren(overgo.failure(err)); run.disabled = true; return; }
        // A task is served by every model activated for it: the recipe is the
        // identity, the model's name the label, and a refused one says why.
        for (const capability of capabilities) {
          capabilitySelect.appendChild(el("option", {
            value: capability.recipe, disabled: !!capability.refusal, title: capability.refusal || "",
            text: capability.task + " / " + (capability.name || fmt.shortID(capability.recipe)) + overgo.unverified(capability) + (capability.refusal ? " (" + capability.refusal + ")" : ""),
          }));
        }
        let fields = new Map();
        function selected() { return capabilities.find((item) => item.recipe === capabilitySelect.value); }
        // The capability's last entries return with it; its presets sit beside the controls.
        function renderControls() {
          const capability = selected();
          fields = overgo.controlInputs(controls, capability ? capability.controls : []);
          placement.textContent = [capability?.host && "Runs on the host processor: it needs no device memory.", capability?.note].filter(Boolean).join(" ");
          placement.hidden = !placement.textContent;
          if (!capability) return;
          overgo.rememberedControls.apply(fields, overgo.rememberedControls.last(capability.recipe));
          if (fields.size) controls.appendChild(overgo.presetControls(capability.recipe, fields));
        }
        capabilitySelect.addEventListener("change", renderControls);
        renderControls();
        if (definition.preview) definition.preview(previewHost, () => ({ task: selected().task, recipe: selected().recipe, input: overgo.controlValues(fields).input }), overgo);
        // The size is asked of the command route, which sizes what it admits: it shows once every required entry is in, is
        // asked again on each change, and a submission the writer would refuse shows its refusal here, before any run.
        if (definition.sized) {
          const sized = overgo.read(sizeHost, async (signal) => {
            const capability = selected();
            const { input, missing, refusal } = overgo.controlValues(fields);
            if (!capability || missing.size || refusal) return {};
            return api.post("/" + definition.scope + "/command", { task: capability.task, recipe: capability.recipe, input }, { signal });
          }, (answer) => sizeHost.replaceChildren(...(answer.bytes ? [el("div", { class: "note", role: "status", "aria-label": "output size", text: "Writes " + fmt.bytes(answer.bytes) + " (" + fmt.grouped(answer.bytes) + " bytes)" })] : [])), { loading: "" });
          for (const [source, events] of [[controls, ["input", "change"]], [capabilitySelect, ["change"]]]) for (const name of events) source.addEventListener(name, () => sized());
          sized();
        }

        let operation = null;
        cancel.addEventListener("click", async () => {
          if (operation) {
            cancel.disabled = true;
            try { await overgo.cancelOperation(operation); } catch (err) { cancel.disabled = false; status.replaceChildren(overgo.failure(err)); }
          }
        });
        function renderOperation(current) {
          const suffix = current.run ? " / " + fmt.shortID(current.run) : "";
          status.textContent = (definition.scope === 'generation' && current.text || current.state) + suffix + (current.failure ? " / " + current.failure : "");
          const total = current.progress && current.progress.total;
          const completed = current.progress ? current.progress.completed : 0;
          progress.hidden = !total;
          if (total) { progress.max = total; progress.value = completed; }
          // The step reached shows as it is reported, and the remainder projects the server's elapsed wall over the steps
          // left, as of the last step, where the server has timed one.
          const elapsed = current.progress && current.progress.elapsed_ms;
          remaining.textContent = total && completed && !terminal(current.state) ?
            "step " + completed + " of " + total + (elapsed ? " · about " + fmt.duration((total - completed) * elapsed / completed) + " remaining" : "") : "";
          const drawn = (current.series || []).filter((series) => definition.trend && definition.trend(series.name) && series.values.length > 1);
          trends.replaceChildren(...drawn.map((series) => el("section", { class: "evidence-block", "data-trend": series.name },
            el("div", { class: "section-title", text: series.name + " · " + series.reports + " steps" }),
            overgo.viz.signedSeries([{ label: series.name, values: series.values, color: "var(--accent)" }]))));
          const table = overgo.table(["measurement", "value"], (current.metrics || []).map((metric) => [metric.name, String(metric.value) + (metric.unit ? " " + metric.unit : "")]), "metric-grid");
          metrics.replaceChildren(...((current.metrics || []).length ? [table] : []));
          // What a running operation reports to be read while it goes (definition.renderLive) stands where its evidence will.
          if (definition.renderLive) definition.renderLive(evidence, current, overgo);
        }
        // follow: the operation's live state until it ends, then its evidence, with the input this page sent it, where it
        // did (a run followed again after a reload has none here).
        async function follow(id, sent) {
          operation = id;
          run.disabled = true;
          cancel.disabled = false;
          cancel.hidden = false;
          status.textContent = "running / " + fmt.shortID(operation);
          try {
            const completed = await overgo.waitOperation(operation, renderOperation);
            overgo.runNotice.ended(overgo.title, completed.state);
            if (completed && completed.state === "completed" && definition.renderEvidence) await definition.renderEvidence(evidence, completed, overgo, sent);
          } catch (err) { status.replaceChildren(overgo.failure(err)); } finally { operation = null; run.disabled = false; cancel.disabled = false; cancel.hidden = true; }
        }
        // submission: the selected capability and its entries, or null after focusing the first missing one.
        function submission() {
          const capability = selected();
          if (!capability) return null;
          const { input, missing, refused, refusal } = overgo.controlValues(fields);
          if (missing.size) { const [name] = missing; fields.get(name).input.focus(); status.textContent = name + " is required"; return null; }
          if (refusal) { fields.get(refused).input.focus(); status.textContent = refusal; return null; }
          return { capability, input };
        }
        controls.addEventListener("input", () => commands.replaceChildren());
        capabilitySelect.addEventListener("change", () => commands.replaceChildren());
        command.addEventListener("click", async () => {
          const submitted = submission();
          if (!submitted) return;
          try {
            const answer = await api.post("/" + definition.scope + "/command", { task: submitted.capability.task, recipe: submitted.capability.recipe, input: submitted.input });
            // The server quotes for a POSIX shell, which a Windows host reaches through Git Bash or WSL, so the page says so.
            commands.replaceChildren(...answer.commands.map((item) => el("div", { class: "workflow-command" },
              el("span", { class: "tag", text: item.kind === "cli" ? "command line" : "HTTP" }), el("pre", { class: "mono", text: item.text }), overgo.copyButton(item.text))),
              ...(answer.commands.length ? [el("p", { class: "note", text: "Written for a POSIX shell: bash or zsh, or Git Bash or WSL on Windows." })] : []));
          } catch (err) { commands.replaceChildren(overgo.failure(err)); }
        });
        run.addEventListener("click", async () => {
          const submitted = submission();
          if (!submitted) return;
          const { capability, input } = submitted;
          if (definition.longRun) overgo.runNotice.ask();
          run.disabled = true;
          evidence.replaceChildren();
          let accepted;
          try {
            accepted = await api.submit("/" + definition.scope + "/run", { task: capability.task, recipe: capability.recipe, input });
            overgo.rememberedControls.remember(capability.recipe, overgo.rememberedControls.values(fields));
          } catch (err) { status.replaceChildren(overgo.failure(err)); run.disabled = false; return; }
          await follow(accepted.operation, input);
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
