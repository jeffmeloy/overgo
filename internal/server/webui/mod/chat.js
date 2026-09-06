/* Chat: the conversation the server owns. Every turn is a stored response
   chained through previous_response_id; the rail's selection resumes a
   chain from its record, a turn cut off mid-stream reattaches through
   /interactions/follow, and defaults come from the capability document. */
(function () {
  "use strict";
  const INFLIGHT_STORAGE = "overgo.inflight"; // the response id of a turn this page was streaming

  // inspectTurn: the side panel over one assistant turn: its run record, then any inspector embedded over it.
  document.addEventListener("keydown", (event) => { // Escape closes the inspector and stops a running turn from anywhere on the page
    const aside = document.getElementById("inspector");
    if (event.key !== "Escape") return;
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
      const { el, clear, fmt } = overgo;
      clear(panel);
      const capabilities = overgo.capabilities();
      if (!capabilities) { panel.appendChild(overgo.errorBanner("the served model declares no capabilities yet")); return; }
      const modelID = capabilities.id;
      const params = capabilities.generation;
      const contextLength = capabilities.context_length;
      let controller = null;
      let lastResponseID = "";

      // Agent mode: the active definitions the store holds; a turn runs the same thread through /agents/chat
      // and the shared tool-step surface runs the session's tool steps inline.
      let agents = [];
      let tools = [];
      try { agents = (await overgo.api.get("/agents")).filter((item) => item.state === "active"); } catch (_) { /* no agent runtime */ }
      if (agents.length) tools = (await overgo.api.get("/agent/tools")).tools || [];
      const agentPicker = el("select", { class: "text", style: "width:auto", "aria-label": "agent" }, ...agents.map((item) => el("option", { value: item.name, text: "agent " + item.name })));
      const agentHost = el("div", { class: "agent-session" });
      agentHost.hidden = true;
      const agentSession = "front-" + new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
      const histories = new Map();

      const system = el("textarea", { class: "text", placeholder: "system prompt (optional)", style: "min-height:52px" });
      const facts = el("div", { class: "statgrid", "aria-label": "context meter" });
      const temperature = el("input", { class: "keyfield", type: "number", value: params.temperature, style: "width:80px" });
      const maxTokens = el("input", { class: "keyfield", type: "number", value: params.max_tokens, max: contextLength, style: "width:90px" });
      const reset = el("button", { class: "btn alt", onclick: () => overgo.openConversation(null) }, "new");

      panel.append(el("details", { style: "margin-bottom:10px" }, el("summary", { class: "note" }, "system prompt"), system), facts);
      // artifactField: the mode's artifact-typed control, if any. A media card re-enters the composer as the next
      // turn's attachment (refused or accepted by the served capability) or, in such a mode, as the control's stored id;
      // a fresh attachment in such a mode stores through the intake route and fills the control.
      const artifactField = () => [...generation.fields.values()].find((field) => field.control.type === "artifact");
      const thread = overgo.thread(panel, { reuse: (file, artifact) => { const field = artifactField(); if (field && artifact) field.input.value = artifact; else composer.addFile(file); },
        marker: capabilities.remote ? "remote" : "" });
      overgo.stopTurn = () => { if (controller) controller.abort(); }; // the Escape key's stop, the composer's stop control's too
      const composer = overgo.composer(panel, {
        onSubmit: submit,
        onStop: overgo.stopTurn,
        takesAny: () => !!artifactField(),
        intake: (file) => { const field = artifactField(); return field ? overgo.api.upload("/artifacts/intake", file).then((stored) => (field.input.value = stored.id)) : null; },
        modes: (capabilities.modes || []).filter((mode) => mode.enabled), // the served recipe declares agent mode with the rest
        onMode: (mode) => { agentHost.hidden = mode !== "agent"; renderMode(mode); },
        controls: [reset, el("span", { class: "note", text: "temp" }), temperature, el("span", { class: "note", text: "max tokens" }), maxTokens],
      });
      panel.insertBefore(agentHost, composer.element);

      // A generation mode renders what its capability declares: the models
      // activated for the task (a refused one says why) and the request's
      // controls, read from the generation capabilities, never from a list
      // typed here; the message body feeds the declared text control.
      const generation = { capabilities: null, capability: null, fields: new Map() };
      async function renderMode(mode) {
        generation.capability = null;
        composer.modeHost.replaceChildren();
        if (!mode || mode === "chat" || mode === "agent") return;
        try {
          if (!generation.capabilities) generation.capabilities = await overgo.api.get("/generation/capabilities");
        } catch (err) { composer.modeHost.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); return; }
        const declared = generation.capabilities.filter((capability) => capability.task === mode);
        if (!declared.length) return;
        const controlsHost = el("span", { class: "row" });
        const picker = el("select", { class: "text", style: "width:auto", "aria-label": "generation model" }, ...declared.map((capability) => el("option", {
          value: capability.recipe, text: capability.name || fmt.shortID(capability.recipe), disabled: !!capability.refusal, title: capability.refusal || "" })));
        const select = () => {
          generation.capability = declared.find((capability) => capability.recipe === picker.value && !capability.refusal) || null;
          const typed = generation.capability ? generation.capability.controls.filter((control) => control !== overgo.bodyControl(generation.capability.controls)) : [];
          generation.fields = overgo.controlInputs(controlsHost, typed);
        };
        picker.addEventListener("change", select);
        composer.modeHost.append(picker, controlsHost);
        select();
      }
      const toolSurface = agents.length ? overgo.toolStep(agentHost, {
        agent: () => agentPicker.value, session: () => agentSession, thread: () => thread, controls: [agentPicker],
        onError: (err) => thread.errorRow(overgo.friendlyError(err)),
      }) : null;
      if (toolSurface) { toolSurface.setAgent(agents[0], tools); agentPicker.addEventListener("change", () => toolSurface.setAgent(agents.find((item) => item.name === agentPicker.value), tools)); }

      // The empty conversation is the getting-started card.
      const served = overgo.servedModel();
      const declared = Object.keys(capabilities.modalities || {}).filter((kind) => capabilities.modalities[kind]).concat((capabilities.modes || []).filter((mode) => mode.enabled).map((mode) => mode.label)).concat(capabilities.remote ? ["remote"] : []);
      const welcome = el("div", { class: "card front-empty" },
        el("h2", { text: capabilities.name || modelID }),
        el("div", { class: "note", text: (served && overgo.evidenceLine(served)) || "no committed evidence yet" }),
        el("div", null, ...declared.map((label) => el("span", { class: "tag", text: label }))),
        el("div", { class: "starters" }, el("button", { class: "btn", text: "Ask a question", onclick: () => composer.input.focus() }), el("button", { class: "btn alt", text: "Attach a file", onclick: () => composer.openPicker() }),
          el("button", { class: "btn alt", text: "Switch model", onclick: () => document.getElementById("model-pill").click() })));
      panel.insertBefore(welcome, thread.node);

      function renderFacts(inputTokens, usage, timings) {
        const cards = [];
        if (inputTokens != null) {
          cards.push(overgo.stat("Input", fmt.grouped(inputTokens), "tokens"));
          if (contextLength != null) cards.push(overgo.stat("Available", fmt.grouped(Number(contextLength) - Number(inputTokens)), "tokens"), overgo.stat("Context ratio", inputTokens + " / " + contextLength));
        }
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
            if (event.type === "created" || event.type === "done") latest = event.id || latest;
            if (event.type === "created") { try { localStorage.setItem(INFLIGHT_STORAGE, event.id); } catch (_) { /* storage unavailable */ } }
            yield event;
          }
        }
        const terminal = await thread.consume(noting(overgo.streams.responses(response)), assistant);
        if (latest) lastResponseID = latest;
        if (latest && assistant) { assistant.response = latest; thread.renderMessage(assistant, false); }
        try { localStorage.removeItem(INFLIGHT_STORAGE); } catch (_) { /* storage unavailable */ }
        renderFacts(inputTokens, terminal.usage, terminal.timings);
        overgo.refreshConversations();
      }

      async function submit(text, attachments, mode) {
        if (controller) return;
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
            const selection = generation.capability ? { capability: generation.capability, fields: generation.fields } : null;
            await thread.consume(overgo.generate(mode, text, parts, controller.signal, selection));
            return;
          }
          assistant = thread.add("assistant", "");
          const request = { model: modelID, input: responsesInput(text, parts), stream: true, store: true };
          if (lastResponseID) request.previous_response_id = lastResponseID;
          if (system.value.trim()) request.instructions = system.value.trim();
          if (temperature.value !== "") request.temperature = Number(temperature.value);
          if (maxTokens.value !== "") request.max_output_tokens = Number(maxTokens.value);
          // Remote model: provider tokenizes; count route refuses -> meter stays empty.
          const count = await overgo.api.post("/v1/responses/input_tokens", request, { signal: controller.signal }).catch(() => null);
          renderFacts(count ? count.input_tokens : null, null, null);
          await consumeTurn(await streamTurn("/v1/responses", request, "POST"), assistant, count ? count.input_tokens : null);
        } catch (err) {
          if (err.name === "AbortError" && assistant) {
            assistant.content += (assistant.content ? "\n" : "") + "[stopped]";
            thread.renderMessage(assistant, false);
          } else thread.errorRow(err.name === "AbortError" ? "cancelled" : overgo.friendlyError(err));
        } finally {
          controller = null;
          composer.setBusy(false);
        }
      }

      // Resume the rail selection from its record, or reattach to a turn this page was streaming.
      const selected = overgo.conversation();
      let inflight = "";
      try { inflight = localStorage.getItem(INFLIGHT_STORAGE) || ""; } catch (_) { /* storage unavailable */ }
      if (selected) {
        try {
          const chain = await overgo.api.get("/interactions/messages?response=" + encodeURIComponent(selected.latest));
          welcome.remove();
          for (const message of chain.messages || []) {
            if (message.role !== "user" && message.role !== "assistant") continue;
            const shown = thread.add(message.role, message.content);
            shown.response = message.response;
            thread.renderMessage(shown, false);
          }
          lastResponseID = chain.response;
        } catch (err) { thread.errorRow(overgo.friendlyError(err)); }
      }
      if (inflight) {
        welcome.remove();
        controller = new AbortController();
        composer.setBusy(true);
        try {
          await consumeTurn(await streamTurn("/interactions/follow?response=" + encodeURIComponent(inflight), null, "GET"), thread.add("assistant", ""), null);
        } catch (err) { thread.errorRow(overgo.friendlyError(err)); } finally { controller = null; composer.setBusy(false); }
      }
    },
  });
})();
