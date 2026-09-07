/* Chat: the conversation the server owns. Every turn is a stored response
   chained through previous_response_id; the rail's selection resumes a
   chain from its record, a turn cut off mid-stream reattaches through
   /interactions/follow, and defaults come from the capability document. */
(function () {
  "use strict";
  const INFLIGHT_STORAGE = "overgo.inflight"; // the response id of a turn this page was streaming
  let disposeChat = () => {};

  // inspectTurn: the side panel over one assistant turn: its run record, then any inspector embedded over it.
  document.addEventListener("keydown", (event) => { // Escape closes the inspector and stops a running turn from anywhere on the page
    const aside = document.getElementById("inspector");
    if (event.key !== "Escape") return;
    if (event.defaultPrevented || document.querySelector("dialog[open]")) return;
    if (aside && !aside.hidden) { aside.hidden = true; aside.replaceChildren(); }
    if (window.overgo.stopTurn) window.overgo.stopTurn();
  });
  window.overgo.inspectTurn = async function (responseID) {
    const overgo = window.overgo;
    const { el } = overgo;
    const aside = document.getElementById("inspector");
    const body = el("div");
    aside.hidden = false;
    const close = el("button", { class: "btn alt", text: "×", "aria-label": "close the inspector", onclick: () => { aside.hidden = true; aside.replaceChildren(); } });
    aside.replaceChildren(el("div", { class: "row" }, el("strong", { text: "Inspect turn" }), el("span", { class: "grow" }), close), body);
    close.focus();
    let record;
    try { record = await overgo.api.get("/interactions/inspect?response=" + encodeURIComponent(responseID)); }
    catch (err) { body.appendChild(overgo.errorBanner(overgo.friendlyError(err))); return; }
    const cards = [overgo.stat("Status", record.status, record.statuses.join(" · "))];
    if (record.failure) cards.push(overgo.stat("Failure", record.failure));
    if (record.timings) cards.push(overgo.stat("Prefill", Number(record.timings.prompt_per_second).toFixed(2), "tok/s"), overgo.stat("Decode", Number(record.timings.predicted_per_second).toFixed(2), "tok/s"));
    const links = ["model", "recipe", "trace", "receipt", "operation", "run"].filter((name) => record[name]).map((name) => el("span", {}, name + " ", overgo.artifactLink(record[name])));
    const host = el("div");
    const seed = { prompt: (record.prompt || "") + (record.completion ? "\n" + record.completion : "") };
    const inspectors = [["lens", "Logits", seed], ["states", "States", seed], ["attention", "Attention", seed], ["model", "Model"], ["vocab", "Vocabulary"], ["tensors", "Tensors"]];
    body.append(el("div", { class: "statgrid" }, ...cards), el("div", { class: "row artifact-links" }, ...links),
      el("div", { class: "row" }, ...inspectors.map(([id, label, given]) => el("button", { class: "btn alt", text: label, onclick: () => overgo.embed(id, host, given) }))), host);
  };

  window.overgo.registerTab({
    id: "chat",
    async mount(panel, overgo) {
      disposeChat();
      let disposed = false;
      let controller = null;
      const selected = overgo.conversation();
      let conversationRoot = selected ? selected.root : "";
      disposeChat = () => { disposed = true; if (controller) controller.abort(); };
      const { el, clear, fmt } = overgo;
      clear(panel);
      const capabilities = overgo.capabilities();
      if (!capabilities) { panel.appendChild(overgo.errorBanner("the served model declares no capabilities yet")); return; }
      const modelID = capabilities.id;
      const served = overgo.servedModel();
      const otherModel = selected && selected.model && served && selected.model !== served.model;
      const params = capabilities.generation;
      const contextLength = capabilities.context_length;
      let lastResponseID = "";

      // Agent mode: the active definitions the store holds; a turn runs the same thread through /agents/chat
      // and the shared tool-step surface runs the session's tool steps inline.
      let agents = [];
      let tools = [];
      try { agents = (await overgo.api.get("/agents")).filter((item) => item.state === "active"); } catch (_) { /* no agent runtime */ }
      if (agents.length) tools = (await overgo.api.get("/agent/tools")).tools || [];
      if (disposed) return;
      const agentPicker = el("select", { class: "text w-auto", "aria-label": "agent" }, ...agents.map((item) => el("option", { value: item.name, text: "agent " + item.name })));
      const agentHost = el("div", { class: "agent-session" });
      agentHost.hidden = true;
      const agentSession = "front-" + new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
      const histories = new Map();

      const system = el("textarea", { class: "text system-prompt", placeholder: "system prompt (optional)" });
      const facts = el("div", { class: "statgrid", "aria-label": "context meter" });
      const temperature = el("input", { class: "keyfield w-80", type: "number", value: params.temperature, "aria-label": "temperature" });
      const maxTokens = el("input", { class: "keyfield w-90", type: "number", value: params.max_tokens, max: contextLength, "aria-label": "max tokens" });

      const settings = document.getElementById("conversation-settings");
      settings.replaceChildren(el("section", { class: "settings-section" }, el("h3", { text: "Conversation" }),
        el("label", { class: "setting-field" }, "System prompt", system),
        el("div", { class: "settings-fields" }, el("label", { class: "setting-field" }, "Temperature", temperature), el("label", { class: "setting-field" }, "Maximum output tokens", maxTokens)),
        el("details", {}, el("summary", { text: "Last response usage" }), facts)));
      // artifactField: the mode's artifact-typed control, if any. A media card re-enters the composer as the next
      // turn's attachment (refused or accepted by the served capability) or, in such a mode, as the control's stored id;
      // a fresh attachment in such a mode stores through the intake route and fills the control.
      // A file goes to the slot of its kind (the slot's declared media), else to the first slot.
      const artifactField = (file) => { const slots = [...generation.fields.values()].filter((field) => field.control.type === "artifact");
        return slots.find((field) => file && field.control.media && file.type.startsWith(field.control.media + "/")) || slots[0]; };
      const thread = overgo.thread(panel, { reuse: (file, artifact) => { const field = artifactField(file); if (field && artifact) field.input.value = artifact; else composer.addFile(file); },
        replay: replayRecord, lineage: showLineage, marker: capabilities.remote ? "remote" : "" });
      // showLineage: what the store records around a card's artifact (the runs that made it, with their
      // request and inputs; the runs that used it, with their outputs) and, as next steps, every active
      // capability whose declared slot takes the artifact's kind, one click opening that mode with it.
      async function showLineage(event, card) {
        let lineage;
        try { lineage = await overgo.api.get("/artifacts/lineage?id=" + encodeURIComponent(event.artifact)); }
        catch (err) { card.appendChild(overgo.errorBanner(overgo.friendlyError(err))); return; }
        const runLine = (verb, run, items) => el("div", {}, verb + " ", overgo.artifactLink(run.run, "run " + fmt.shortID(run.run)), " " + run.outcome + " · ",
          ...items.map((item) => el("span", {}, item.name, overgo.artifactLink(item.id), " ")));
        const inputName = (input) => { const schema = input.schema || ""; if (schema.endsWith("-input.v1")) return "request "; return schema.startsWith("overgo/prompt-enhancement/") ? "prompt " : ""; };
        const block = el("div", { class: "record lineage" },
          ...lineage.producers.map((run) => runLine("made by", run, run.inputs.map((input) => ({ id: input.id, name: inputName(input) })))),
          ...lineage.consumers.map((run) => runLine("used by", run, run.outputs.map((id) => ({ id, name: "" })))),
          lineage.producers.length + lineage.consumers.length ? null : el("span", { class: "note", text: "no run recorded around it" }),
          el("div", { class: "row", "aria-label": "next steps" }, ...lineage.next.map((step) => el("button", { class: "chip", text: step.name + " · " + step.label, onclick: () => openStep(step, event.artifact) }))));
        card.querySelector(".lineage") ? card.querySelector(".lineage").replaceWith(block) : card.appendChild(block);
      }
      // openStep: the composer in the step's mode with its model picked and the artifact in the declared slot.
      // requestOf: the request document the run that made an artifact cites (its input under an input
      // schema; a run's inputs stand in identity order), read from the artifact's lineage.
      async function requestOf(artifact, run) {
        const lineage = await overgo.api.get("/artifacts/lineage?id=" + encodeURIComponent(artifact));
        const made = lineage.producers.find((item) => item.run === run) || lineage.producers[0];
        const input = made && made.inputs.find((item) => (item.schema || "").endsWith("-input.v1"));
        return input && overgo.api.get("/artifacts/content?id=" + encodeURIComponent(input.id));
      }
      async function openStep(step, artifact) {
        await composer.setMode(step.task);
        if (generation.picker) { generation.picker.value = step.recipe; generation.picker.dispatchEvent(new Event("change")); }
        const field = generation.fields.get(step.control);
        if (field) { field.input.value = artifact; field.input.dispatchEvent(new Event("change", { bubbles: true })); }
        composer.input.focus();
      }
      // replayRecord: a card's stored request (the run's input document) resubmitted to the capability
      // that made it, unchanged or with a fresh seed; page state plays no part in the request.
      async function replayRecord(event, label) {
        if (controller) return;
        controller = new AbortController();
        composer.setBusy(true);
        try {
          if (!generation.capabilities) generation.capabilities = await overgo.api.get("/generation/capabilities");
          const run = await overgo.api.get("/runs?id=" + encodeURIComponent(event.run));
          const capability = generation.capabilities.find((item) => item.recipe === run.recipe && !item.refusal);
          if (!capability) throw new Error("the capability that made it is no longer active");
          const request = await requestOf(event.artifact, run.id);
          if (!request) throw new Error("the run records no request");
          if (label === "vary") {
            if (!capability.controls.some((control) => control.name === "seed")) throw new Error("the request declares no seed to vary");
            request.seed = crypto.getRandomValues(new Uint32Array(1))[0];
          }
          thread.add("user", label + " " + fmt.shortID(event.artifact));
          await thread.consume(overgo.streams.replay(capability, request, controller.signal, event.artifact, label));
        } catch (err) { thread.errorRow(err.name === "AbortError" ? "cancelled" : overgo.friendlyError(err)); }
        finally { controller = null; composer.setBusy(false); }
      }
      overgo.stopTurn = () => {
        if (controller) controller.abort();
        try { sessionStorage.removeItem(INFLIGHT_STORAGE); } catch (_) { /* storage unavailable */ }
      };
      const composer = overgo.composer(panel, {
        onSubmit: submit,
        onStop: overgo.stopTurn,
        takesAny: () => !!artifactField(),
        intake: (file) => { const field = artifactField(file); return field ? overgo.api.upload("/artifacts/intake", file).then((stored) => { field.input.value = stored.id; field.input.dispatchEvent(new Event("intake")); return stored.id; }) : null; },
        modes: (capabilities.modes || []).filter((mode) => mode.enabled), // the served recipe declares agent mode with the rest
        onMode: (mode) => { agentHost.hidden = mode !== "agent"; return renderMode(mode); },
        controls: [],
      });
      settings.firstChild.appendChild(el("button", { class: "btn alt", text: "Enhance current prompt", onclick: () => { document.getElementById("settings-dialog").close(); enhancePrompt(); } }));
      panel.insertBefore(agentHost, composer.element);
      // Prompt enhancement: the served model rewrites the typed prompt under the server's fixed
      // instruction; the rewrite stands beside the original until accepted, and an accepted rewrite
      // makes the stored enhancement record the next generation run's source.
      const enhancement = { record: "", enhanced: "" };
      const enhanceHost = el("div", { class: "enhancement" });
      panel.insertBefore(enhanceHost, composer.element);
      async function enhancePrompt() {
        const prompt = composer.input.value.trim();
        if (!prompt) { thread.errorRow("type a prompt to enhance"); return; }
        enhanceHost.replaceChildren(el("span", { class: "note", text: "rewriting through " + modelID }));
        try {
          const answer = await overgo.api.post("/generation/enhance", { prompt });
          const accept = el("button", { class: "btn", text: "accept", onclick: () => {
            composer.input.value = answer.enhanced; enhancement.record = answer.record || ""; enhancement.enhanced = answer.enhanced; enhanceHost.replaceChildren(); composer.input.focus(); } });
          const keep = el("button", { class: "btn alt", text: "keep original", onclick: () => { enhanceHost.replaceChildren(); composer.input.focus(); } });
          enhanceHost.replaceChildren(el("div", { class: "card" }, el("div", { class: "note", text: "original: " + answer.original }),
            el("div", { class: "rewrite", text: answer.enhanced }), el("div", { class: "row" }, accept, keep)));
        } catch (err) { enhanceHost.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }

      // Generation modes project declared capabilities; transitions discard old slots.
      const generation = { capabilities: null, capability: null, fields: new Map() };
      const galleryLimit = 12; // the newest outputs a mode's gallery rail lists
      async function renderMode(mode) {
        generation.capability = null;
        generation.fields.clear(); generation.picker = null;
        composer.modeHost.replaceChildren();
        if (!mode || mode === "chat" || mode === "agent") return;
        try {
          if (!generation.capabilities) generation.capabilities = await overgo.api.get("/generation/capabilities");
        } catch (err) { composer.modeHost.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); return; }
        if (composer.mode() !== mode) return;
        const declared = generation.capabilities.filter((capability) => capability.task === mode);
        if (!declared.length) return;
        const controlsHost = el("span", { class: "row" });
        const picker = el("select", { class: "text w-auto", "aria-label": "generation model" }, ...declared.map((capability) => el("option", {
          value: capability.recipe, text: capability.name || fmt.shortID(capability.recipe), disabled: !!capability.refusal, title: capability.refusal || "" })));
        // The gallery rail: the store's outputs of the mode's kind newest first, each thumb named by the
        // capability that made it; "this model" keeps the picked one's. A thumb opens the record.
        const rail = el("div", { class: "intake-strip gallery", "aria-label": "recent " + overgo.outputKind(mode) + " outputs" });
        const only = el("input", { type: "checkbox", "aria-label": "this model only" });
        const filterRail = () => { for (const thumb of rail.children) thumb.hidden = only.checked && thumb.dataset.recipe !== picker.value; };
        const select = () => {
          generation.capability = declared.find((capability) => capability.recipe === picker.value && !capability.refusal) || null;
          const typed = generation.capability ? generation.capability.controls.filter((control) => control !== overgo.bodyControl(generation.capability.controls)) : [];
          generation.fields = overgo.controlInputs(controlsHost, typed);
          filterRail();
        };
        picker.addEventListener("change", select);
        generation.picker = picker;
        only.addEventListener("change", filterRail);
        composer.modeHost.append(picker, controlsHost, el("label", { class: "chip" }, only, " this model"), rail);
        select();
        galleryRail(rail, overgo.outputKind(mode)).then(filterRail);
      }
      async function galleryRail(rail, kind) {
        try {
          const listed = await overgo.api.get("/artifacts?kind=output&newest=1&media=" + encodeURIComponent(kind + "/") + "&limit=" + galleryLimit);
          for (const item of (listed.artifacts || []).filter((item) => item.payload && item.producers.length)) {
            const run = await overgo.api.get("/runs?id=" + encodeURIComponent(item.producers[0]));
            const made = generation.capabilities.find((capability) => capability.recipe === run.recipe);
            const url = "/artifacts/content?id=" + encodeURIComponent(item.descriptor.id);
            const name = made ? made.name : fmt.shortID(run.recipe);
            rail.appendChild(el("button", { class: "intake-thumb", type: "button", "data-id": item.descriptor.id, "data-recipe": run.recipe, title: name,
              "aria-label": "open " + fmt.shortID(item.descriptor.id), onclick: () => openRecord(item, run, name, url) },
              kind === "image" ? el("img", { src: url, alt: "" }) : el("span", { class: "mono", text: item.descriptor.media_type })));
          }
        } catch (err) { rail.replaceChildren(el("span", { class: "note", text: overgo.friendlyError(err) })); }
      }
      // openRecord: a gallery output as a media card (use as input, stored-as) with the record that made
      // it: the run's request document as control/value rows, the run, and a download of the bytes.
      async function openRecord(item, run, name, url) {
        const card = thread.mediaCard({ kind: overgo.mediaKind(item.descriptor.media_type), url, artifact: item.descriptor.id, mime: item.descriptor.media_type, bytes: item.descriptor.size, caption: name, run: run.id });
        let request = {};
        try { request = await requestOf(item.descriptor.id, run.id) || {}; } catch (err) { card.appendChild(overgo.errorBanner(overgo.friendlyError(err))); }
        card.appendChild(el("div", { class: "record" }, overgo.table(["control", "value"], Object.entries(request).map(([control, value]) => [control, String(value)])),
          el("div", { class: "row" }, overgo.artifactLink(run.id, "run " + fmt.shortID(run.id)), el("a", { class: "btn alt", href: url, download: "", text: "download" }))));
      }
      const toolSurface = agents.length ? overgo.toolStep(agentHost, {
        agent: () => agentPicker.value, session: () => agentSession, thread: () => thread, controls: [agentPicker],
        onError: (err) => thread.errorRow(overgo.friendlyError(err)),
      }) : null;
      if (toolSurface) { toolSurface.setAgent(agents[0], tools); agentPicker.addEventListener("change", () => toolSurface.setAgent(agents.find((item) => item.name === agentPicker.value), tools)); }

      // The empty conversation offers direct starting actions.
      const welcome = el("div", { class: "card front-empty" },
        el("h2", { text: "What would you like to work on?" }),
        el("div", { class: "starters" }, el("button", { class: "btn", text: "Ask a question", onclick: () => composer.input.focus() }), el("button", { class: "btn alt", text: "Attach a file", onclick: () => composer.openPicker() }),
          // With nothing catalogued the same control reads as the way in; the picker names the two paths.
          el("button", { class: "btn alt", text: served ? "Switch model" : "Add a model", onclick: () => document.getElementById("model-pill").click() })));
      thread.node.prepend(welcome);

      function renderFacts(inputTokens, usage, timings) {
        const cards = [];
        if (inputTokens == null && usage && usage.prompt_tokens) inputTokens = usage.prompt_tokens; // a hosted turn: the provider's count, the count route having refused
        if (inputTokens != null) { cards.push(overgo.stat("Input", fmt.grouped(inputTokens), "tokens")); if (contextLength != null) cards.push(overgo.stat("Available", fmt.grouped(Number(contextLength) - Number(inputTokens)), "tokens"), overgo.stat("Context ratio", inputTokens + " / " + contextLength)); }
        if (usage && usage.completion_tokens != null) cards.push(overgo.stat("Completion", fmt.grouped(usage.completion_tokens), "tokens"));
        if (timings) {
          cards.push(overgo.stat("Cached", fmt.grouped(timings.cache_n), "tokens"));
          if (timings.prompt_n && timings.prompt_ms > 0) cards.push(overgo.stat("Prefill", Number(timings.prompt_per_second).toFixed(2), "tok/s"));
          if (timings.predicted_n && timings.predicted_ms > 0) cards.push(overgo.stat("Decode", Number(timings.predicted_per_second).toFixed(2), "tok/s"));
          if (Number(timings.prompt_ms) + Number(timings.predicted_ms) > 0) cards.push(overgo.stat("Elapsed", (Number(timings.prompt_ms) + Number(timings.predicted_ms)).toFixed(2), "ms"));
        }
        facts.replaceChildren(...cards);
      }

      // The composer's protocol parts become Responses input parts.
      function responsesInput(text, parts) {
        const content = [{ type: "input_text", text }];
        for (const part of parts) {
          if (part.type === "image_url") content.push({ type: "input_image", image_url: part.image_url.url });
          else if (part.type === "input_audio") content.push({ type: "input_audio", input_audio: part.input_audio });
          else if (part.type === "input_video") content.push({ type: "input_video", input_video: part.input_video });
          else if (part.type === "input_file") content.push(part);
        }
        return [{ role: "user", content: parts.length ? content : text }];
      }

      function streamTurn(path, body, method) { return overgo.api.stream(path, body, { signal: controller.signal, method }); }

      // consumeTurn drives one streamed turn: the created id is noted so a reload can reattach, the completion id becomes latest.
      async function consumeTurn(response, assistant, inputTokens) {
        let latest = "";
        async function* noting(events) {
          for await (const event of events) {
            if (disposed) return;
            if (event.type === "created" || event.type === "done") latest = event.id || latest;
            if (event.type === "created") {
              const root = conversationRoot || (conversationRoot = event.id);
              overgo.rememberConversation({ root, latest: event.id, model: served && served.model });
              try { sessionStorage.setItem(INFLIGHT_STORAGE, JSON.stringify({ root, response: event.id })); } catch (_) { /* storage unavailable */ }
            }
            yield event;
          }
        }
        const terminal = await thread.consume(noting(overgo.streams.responses(response)), assistant);
        if (disposed) return;
        if (latest) lastResponseID = latest;
        if (latest && assistant) { assistant.response = latest; thread.renderMessage(assistant, false); }
        try { sessionStorage.removeItem(INFLIGHT_STORAGE); } catch (_) { /* storage unavailable */ }
        renderFacts(inputTokens, terminal.usage, terminal.timings);
        overgo.refreshConversations();
      }

      async function submit(text, attachments, mode) {
        if (disposed || controller || otherModel) return;
        welcome.remove();
        const parts = composer.attachmentParts();
        composer.clearInput();
        thread.add("user", overgo.userLine(text, attachments));
        composer.clearAttachments();
        controller = new AbortController();
        composer.setBusy(true);
        let assistant = null;
        try {
          // A mode beyond chat is one generation over the shared dispatch; its
          // media lands in this thread as artifacts with their provenance.
          if (mode === "agent") {
            const history = (histories.get(agentPicker.value) || []).concat([{ role: "user", content: parts.length ? [{ type: "text", text }, ...parts] : text }]);
            const result = await overgo.api.post("/agents/chat", { agent: agentPicker.value, messages: history }, { signal: controller.signal });
            assistant = thread.add("assistant", "");
            await thread.consume(overgo.streams.reply(result), assistant);
            if (assistant.content) history.push({ role: "assistant", content: assistant.content });
            histories.set(agentPicker.value, history);
            return;
          }
          if (mode && mode !== "chat") {
            // An accepted rewrite sent unchanged cites its enhancement record as the run's source.
            const sources = enhancement.record && text === enhancement.enhanced ? [enhancement.record] : [];
            enhancement.record = enhancement.enhanced = "";
            const selection = generation.capability ? { capability: generation.capability, fields: generation.fields, sources } : null;
            await thread.consume(overgo.generate(mode, text, parts, controller.signal, selection));
            return;
          }
          assistant = thread.add("assistant", "");
          const request = { model: modelID, input: responsesInput(text, parts), stream: true, store: true };
          if (lastResponseID) request.previous_response_id = lastResponseID;
          if (system.value.trim()) request.instructions = system.value.trim();
          if (temperature.value !== "") request.temperature = Number(temperature.value);
          if (maxTokens.value !== "") request.max_output_tokens = Number(maxTokens.value);
          const count = await overgo.api.post("/v1/responses/input_tokens", request, { signal: controller.signal }).catch(() => null) /* reviewed: a hosted model's provider tokenizes, the count route refuses, and the meter stays empty by design */;
          renderFacts(count ? count.input_tokens : null, null, null);
          await consumeTurn(await streamTurn("/v1/responses", request, "POST"), assistant, count ? count.input_tokens : null);
        } catch (err) {
          if (err.name === "AbortError" && assistant) {
            assistant.content += (assistant.content ? "\n" : "") + "[stopped]";
            thread.renderMessage(assistant, false);
          } else thread.errorRow(err.name === "AbortError" ? "cancelled" : overgo.friendlyError(err));
        } finally { controller = null; composer.setBusy(false); }
      }

      // Resume the rail selection from its record, or reattach to a turn this page was streaming.
      let saved = null;
      try { saved = JSON.parse(sessionStorage.getItem(INFLIGHT_STORAGE) || "null"); } catch (_) { /* storage unavailable */ }
      const inflight = selected && saved && selected.root === saved.root ? saved.response : "";
      if (selected) {
        composer.setBusy(true);
        try {
          const chain = await overgo.api.get("/interactions/messages?response=" + encodeURIComponent(selected.latest));
          if (disposed) return;
          welcome.remove();
          for (const message of chain.messages || []) {
            if (message.role !== "user" && message.role !== "assistant") continue;
            if (inflight && message.role === "assistant" && message.response === inflight) continue;
            const shown = thread.add(message.role, message.content);
            shown.response = message.response;
            thread.renderMessage(shown, false);
          }
          lastResponseID = chain.response;
        } catch (err) { thread.errorRow(overgo.friendlyError(err)); }
      }
      if (inflight && !disposed && !otherModel) {
        welcome.remove();
        controller = new AbortController();
        composer.setBusy(true);
        try {
          await consumeTurn(await streamTurn("/interactions/follow?response=" + encodeURIComponent(inflight), null, "GET"), thread.add("assistant", ""), null);
        } catch (err) { thread.errorRow(overgo.friendlyError(err)); } finally { controller = null; composer.setBusy(false); }
      }
      composer.setBusy(false);
      if (otherModel && !disposed) {
        composer.setReadOnly(true);
        thread.node.appendChild(el("div", { class: "note", role: "status" }, "This conversation belongs to another model. Choose its original model to continue, or start a new conversation. ",
          el("button", { class: "btn alt", text: "Choose model", onclick: () => document.getElementById("model-pill").click() }),
          el("button", { class: "btn alt", text: "New conversation", onclick: () => overgo.openConversation(null) })));
      }
    },
  });
})();
