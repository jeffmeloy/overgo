/* Chat: the conversation the server owns. Every turn is a stored response
   chained through previous_response_id; the rail's selection resumes a
   chain from its record, a turn cut off mid-stream reattaches through
   /interactions/follow, and defaults come from the capability document. */
(function () {
  "use strict";
  const INFLIGHT_STORAGE = "overgo.inflight"; // the response id of a turn this page was streaming
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

      const system = el("textarea", { class: "text", placeholder: "system prompt (optional)", style: "min-height:52px" });
      const facts = el("div", { class: "statgrid", "aria-label": "context meter" });
      const temperature = el("input", { class: "keyfield", type: "number", value: params.temperature, style: "width:80px" });
      const maxTokens = el("input", { class: "keyfield", type: "number", value: params.max_tokens, max: contextLength, style: "width:90px" });
      const reset = el("button", { class: "btn alt", onclick: () => overgo.openConversation(null) }, "new");

      panel.append(
        el("details", { style: "margin-bottom:10px" }, el("summary", { class: "note" }, "system prompt"), system),
        facts);
      const thread = overgo.thread(panel);
      const composer = overgo.composer(panel, {
        onSubmit: submit,
        onStop: () => { if (controller) controller.abort(); },
        controls: [reset,
          el("span", { class: "note", text: "temp" }), temperature,
          el("span", { class: "note", text: "max tokens" }), maxTokens],
      });

      // The empty conversation is the getting-started card.
      const served = overgo.servedModel();
      const declared = Object.keys(capabilities.modalities || {}).filter((kind) => capabilities.modalities[kind]).concat((capabilities.modes || []).filter((mode) => mode.enabled).map((mode) => mode.label));
      const welcome = el("div", { class: "card front-empty" },
        el("h2", { text: capabilities.name || modelID }),
        el("div", { class: "note", text: (served && overgo.evidenceLine(served)) || "no committed evidence yet" }),
        el("div", null, ...declared.map((label) => el("span", { class: "tag", text: label }))),
        el("div", { class: "starters" },
          el("button", { class: "btn", text: "Ask a question", onclick: () => composer.input.focus() }),
          el("button", { class: "btn alt", text: "Attach a file", onclick: () => composer.openPicker() }),
          el("button", { class: "btn alt", text: "Switch model", onclick: () => document.getElementById("model-pill").click() })));
      panel.insertBefore(welcome, thread.node);

      function renderFacts(inputTokens, usage, timings) {
        const cards = [];
        if (inputTokens != null) {
          cards.push(overgo.stat("Input", fmt.grouped(inputTokens), "tokens"));
          if (contextLength != null) {
            cards.push(overgo.stat("Available", fmt.grouped(Number(contextLength) - Number(inputTokens)), "tokens"));
            cards.push(overgo.stat("Context ratio", inputTokens + " / " + contextLength));
          }
        }
        if (usage && usage.completion_tokens != null) cards.push(overgo.stat("Completion", fmt.grouped(usage.completion_tokens), "tokens"));
        if (timings) {
          cards.push(overgo.stat("Cached", fmt.grouped(timings.cache_n), "tokens"));
          if (timings.prompt_n && timings.prompt_ms > 0) cards.push(overgo.stat("Prefill", Number(timings.prompt_per_second).toFixed(2), "tok/s"));
          if (timings.predicted_n && timings.predicted_ms > 0) cards.push(overgo.stat("Decode", Number(timings.predicted_per_second).toFixed(2), "tok/s"));
          const elapsed = Number(timings.prompt_ms) + Number(timings.predicted_ms);
          if (elapsed > 0) cards.push(overgo.stat("Elapsed", elapsed.toFixed(2), "ms"));
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
        }
        return [{ role: "user", content: parts.length ? content : text }];
      }

      function streamTurn(path, body, method) {
        return overgo.api.stream(path, body, { signal: controller.signal, method });
      }

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
        try { localStorage.removeItem(INFLIGHT_STORAGE); } catch (_) { /* storage unavailable */ }
        renderFacts(inputTokens, terminal.usage, terminal.timings);
        overgo.refreshConversations();
      }

      async function submit(text, attachments) {
        if (controller) return;
        welcome.remove();
        const parts = composer.attachmentParts();
        composer.clearInput();
        thread.add("user", attachments.length ? text + "\n[" + attachments.map((item) => item.kind + ": " + item.name).join(", ") + "]" : text);
        composer.clearAttachments();
        const assistant = thread.add("assistant", "");
        controller = new AbortController();
        composer.setBusy(true);
        try {
          const request = { model: modelID, input: responsesInput(text, parts), stream: true, store: true };
          if (lastResponseID) request.previous_response_id = lastResponseID;
          if (system.value.trim()) request.instructions = system.value.trim();
          if (temperature.value !== "") request.temperature = Number(temperature.value);
          if (maxTokens.value !== "") request.max_output_tokens = Number(maxTokens.value);
          const count = await overgo.api.post("/v1/responses/input_tokens", request, { signal: controller.signal });
          renderFacts(count.input_tokens, null, null);
          await consumeTurn(await streamTurn("/v1/responses", request, "POST"), assistant, count.input_tokens);
        } catch (err) {
          if (err.name === "AbortError") {
            assistant.content += (assistant.content ? "\n" : "") + "[stopped]";
            thread.renderMessage(assistant, false);
          } else thread.errorRow(overgo.friendlyError(err));
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
          for (const message of chain.messages || []) if (message.role === "user" || message.role === "assistant") thread.add(message.role, message.content);
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
