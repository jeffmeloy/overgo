/* Chat: conversation against the served model through /v1/chat/completions,
   over the shared composer and thread (composer.js). Attachments ride the
   served protocol's own content parts: image_url data URLs, input_audio
   base64 WAV, input_video data URLs -- the same parts any API client sends,
   no side channel. Sampling defaults come from the server's /props. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "chat",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);

      let model;
      let properties;
      try {
        [model, properties] = await Promise.all([overgo.modelInfo(), overgo.api.get("/props")]);
      } catch (err) {
        panel.appendChild(overgo.errorBanner(overgo.friendlyError(err)));
        return;
      }
      const modelID = model.model.id;
      const defaults = properties.default_generation_settings;
      const params = defaults.params;
      const contextLength = defaults.n_ctx;
      let controller = null;

      const system = el("textarea", { class: "text", placeholder: "system prompt (optional)", style: "min-height:52px" });
      const facts = el("div", { class: "statgrid" });
      const temperature = el("input", { class: "keyfield", type: "number", value: params.temperature, style: "width:80px" });
      const maxTokens = el("input", { class: "keyfield", type: "number", value: params.max_tokens, max: contextLength, style: "width:90px" });
      const reset = el("button", { class: "btn alt", onclick: clearChat }, "clear");

      panel.append(
        el("details", { style: "margin-bottom:10px" }, el("summary", { class: "note" }, "system prompt"), system),
        facts);
      const thread = overgo.thread(panel);
      const composer = overgo.composer(panel, {
        accept: ["image/png", "image/jpeg", "image/gif", "audio/wav", "video/mp4"],
        onSubmit: submit,
        onStop: () => { if (controller) controller.abort(); },
        controls: [reset,
          el("span", { class: "note", text: "temp" }), temperature,
          el("span", { class: "note", text: "max tokens" }), maxTokens],
      });

      function renderFacts(inputTokens, usage, timings) {
        const cards = [];
        if (inputTokens != null) {
          cards.push(overgo.stat("Input", fmt.grouped(inputTokens), "tokens"));
          if (contextLength != null) {
            cards.push(overgo.stat("Available", fmt.grouped(Number(contextLength) - Number(inputTokens)), "tokens"));
            cards.push(overgo.stat("Context ratio", inputTokens + " / " + contextLength));
          }
        }
        if (usage && usage.completion_tokens != null) {
          cards.push(overgo.stat("Completion", fmt.grouped(usage.completion_tokens), "tokens"));
        }
        if (timings) {
          cards.push(overgo.stat("Cached", fmt.grouped(timings.cache_n), "tokens"));
          if (timings.prompt_n && timings.prompt_ms > 0) {
            cards.push(overgo.stat("Prefill", Number(timings.prompt_per_second).toFixed(2), "tok/s"));
          }
          if (timings.predicted_n && timings.predicted_ms > 0) {
            cards.push(overgo.stat("Decode", Number(timings.predicted_per_second).toFixed(2), "tok/s"));
          }
          const elapsed = Number(timings.prompt_ms) + Number(timings.predicted_ms);
          if (elapsed > 0) cards.push(overgo.stat("Elapsed", elapsed.toFixed(2), "ms"));
        }
        facts.replaceChildren(...cards);
      }

      function clearChat() {
        thread.reset();
        facts.replaceChildren();
      }

      // The request history is the thread's own rows (user and assistant
      // text), so the served model sees exactly what the thread shows.
      function requestMessages(text, parts) {
        const payload = [];
        if (system.value.trim()) payload.push({ role: "system", content: system.value.trim() });
        for (const message of thread.messages) {
          if (message.role === "user" || message.role === "assistant") payload.push({ role: message.role, content: message.content });
        }
        payload.push(parts.length
          ? { role: "user", content: [{ type: "text", text }, ...parts] }
          : { role: "user", content: text });
        return payload;
      }

      async function submit(text, attachments) {
        if (controller) return;
        const parts = composer.attachmentParts();
        const payload = requestMessages(text, parts);
        composer.clearInput();
        const attachedNote = attachments.length
          ? text + "\n[" + attachments.map((item) => item.kind + ": " + item.name).join(", ") + "]"
          : text;
        thread.add("user", attachedNote);
        composer.clearAttachments();
        const assistant = thread.add("assistant", "");
        controller = new AbortController();
        composer.setBusy(true);
        try {
          const count = await overgo.api.post("/v1/chat/completions/input_tokens", {
            model: modelID,
            messages: payload,
          }, { signal: controller.signal });
          renderFacts(count.input_tokens, null, null);
          const request = { model: modelID, messages: payload, stream: true };
          if (temperature.value !== "") request.temperature = Number(temperature.value);
          if (maxTokens.value !== "") request.max_tokens = Number(maxTokens.value);
          const response = await overgo.api.stream("/v1/chat/completions", request, { signal: controller.signal });
          const terminal = await thread.consume(overgo.streams.openai(response), assistant);
          renderFacts(count.input_tokens, terminal.usage, terminal.timings);
        } catch (err) {
          if (err.name === "AbortError") {
            assistant.content += (assistant.content ? "\n" : "") + "[stopped]";
            thread.renderMessage(assistant, false);
          } else {
            thread.errorRow(overgo.friendlyError(err));
          }
        } finally {
          controller = null;
          composer.setBusy(false);
        }
      }
    },
  });
})();
