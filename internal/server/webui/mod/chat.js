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
      const messages = [];
      let controller = null;

      const system = el("textarea", { class: "text", placeholder: "system prompt (optional)", style: "min-height:52px" });
      const log = el("div", { class: "chat-log" });
      const facts = el("div", { class: "statgrid" });
      const input = el("textarea", { class: "text", placeholder: "message (Enter to send, Shift+Enter for newline)" });
      const temperature = el("input", { class: "keyfield", type: "number", value: params.temperature, style: "width:80px" });
      const maxTokens = el("input", { class: "keyfield", type: "number", value: params.max_tokens, max: contextLength, style: "width:90px" });
      const send = el("button", { class: "btn", onclick: submit }, "send");
      const stop = el("button", { class: "btn alt", onclick: abort, style: "display:none" }, "stop");
      const reset = el("button", { class: "btn alt", onclick: clearChat }, "clear");

      // Attachments ride the served protocol's own content parts:
      // image_url data URLs, input_audio base64 WAV, input_video data
      // URLs -- the same parts any API client sends, no side channel.
      const attachments = [];
      const attachmentHost = el("div", { class: "row" });
      const picker = el("input", {
        type: "file", style: "display:none", multiple: true,
        accept: "image/png,image/jpeg,image/gif,audio/wav,video/mp4",
      });
      const attach = el("button", { class: "btn alt", onclick: () => picker.click() }, "attach");
      picker.addEventListener("change", () => {
        for (const file of picker.files) {
          const reader = new FileReader();
          reader.onload = () => {
            const kind = file.type.startsWith("image/") ? "image"
              : file.type.startsWith("audio/") ? "audio" : "video";
            attachments.push({ kind, name: file.name, dataURL: reader.result });
            renderAttachments();
          };
          reader.readAsDataURL(file);
        }
        picker.value = "";
      });

      function renderAttachments() {
        attachmentHost.replaceChildren(...attachments.map((item, index) => {
          const remove = el("button", { class: "btn alt", text: "×" });
          remove.addEventListener("click", () => { attachments.splice(index, 1); renderAttachments(); });
          const preview = item.kind === "image"
            ? el("img", { src: item.dataURL, style: "max-height:48px;max-width:96px" })
            : el("span", { class: "tag", text: item.kind });
          return el("span", { class: "card" }, preview, " " + item.name + " ", remove);
        }));
      }

      function attachmentParts() {
        return attachments.map((item) => {
          if (item.kind === "image") {
            return { type: "image_url", image_url: { url: item.dataURL } };
          }
          if (item.kind === "audio") {
            return { type: "input_audio", input_audio: { data: item.dataURL.split(",").pop(), format: "wav" } };
          }
          return { type: "input_video", input_video: { data: item.dataURL } };
        });
      }

      panel.append(
        el("details", { style: "margin-bottom:10px" }, el("summary", { class: "note" }, "system prompt"), system),
        facts,
        log,
        input,
        attachmentHost,
        el("div", { class: "chat-controls" },
          send, stop, reset, attach, picker,
          el("span", { class: "note", text: "temp" }), temperature,
          el("span", { class: "note", text: "max tokens" }), maxTokens));

      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(); }
      });

      function messageNode(message, streaming) {
        const body = el("div", { class: "body" });
        if (message.role === "assistant" && !streaming && message.content) {
          body.appendChild(overgo.md(message.content));
        } else {
          body.appendChild(document.createTextNode(message.content));
          if (streaming) body.appendChild(el("span", { class: "cursor", text: "|" }));
        }
        const head = el("div", { class: "role" }, message.role);
        if (message.role === "assistant" && !streaming && message.content) {
          head.appendChild(overgo.copyButton(message.content, "copy"));
        }
        return el("div", { class: "msg " + message.role }, head, body);
      }

      function renderLog(streamingCursor) {
        clear(log);
        messages.forEach((message, index) => {
          const streaming = streamingCursor && index === messages.length - 1 && message.role === "assistant";
          log.appendChild(messageNode(message, streaming));
        });
        log.scrollTop = log.scrollHeight;
      }

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
        messages.length = 0;
        renderLog(false);
        facts.replaceChildren();
      }

      function abort() {
        if (controller) controller.abort();
      }

      function requestMessages(text) {
        const payload = [];
        if (system.value.trim()) payload.push({ role: "system", content: system.value.trim() });
        const parts = attachmentParts();
        const user = parts.length
          ? { role: "user", content: [{ type: "text", text }, ...parts] }
          : { role: "user", content: text };
        payload.push(...messages, user);
        return payload;
      }

      async function submit() {
        const text = input.value.trim();
        if (!text || controller) return;
        const payload = requestMessages(text);
        input.value = "";
        const attachedNote = attachments.length
          ? text + "\n[" + attachments.map((item) => item.kind + ": " + item.name).join(", ") + "]"
          : text;
        messages.push({ role: "user", content: attachedNote });
        attachments.length = 0;
        renderAttachments();
        const assistant = { role: "assistant", content: "" };
        messages.push(assistant);
        renderLog(true);

        controller = new AbortController();
        send.disabled = true;
        stop.style.display = "";
        try {
          const count = await overgo.api.post("/v1/chat/completions/input_tokens", {
            model: modelID,
            messages: payload,
          }, { signal: controller.signal });
          renderFacts(count.input_tokens, null, null);
          const request = { model: modelID, messages: payload, stream: true };
          if (temperature.value !== "") request.temperature = Number(temperature.value);
          if (maxTokens.value !== "") request.max_tokens = Number(maxTokens.value);
          const terminal = await consume(await overgo.api.stream(
            "/v1/chat/completions", request, { signal: controller.signal }), assistant);
          renderFacts(count.input_tokens, terminal.usage, terminal.timings);
        } catch (err) {
          if (err.name === "AbortError") {
            assistant.content += (assistant.content ? "\n" : "") + "[stopped]";
          } else {
            assistant.content = "Error: " + overgo.friendlyError(err);
          }
        } finally {
          controller = null;
          send.disabled = false;
          stop.style.display = "none";
          renderLog(false);
        }
      }

      async function consume(response, assistant) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        const terminal = {};
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) return terminal;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;
            const data = trimmed.slice(5).trim();
            if (data === "[DONE]") return terminal;
            let parsed;
            try { parsed = JSON.parse(data); } catch (_) { continue; }
            if (parsed.usage) terminal.usage = parsed.usage;
            if (parsed.timings) terminal.timings = parsed.timings;
            const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
            if (delta && delta.content) {
              assistant.content += delta.content;
              renderLog(true);
            }
          }
        }
      }
    },
  });
})();
