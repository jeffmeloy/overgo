/* Inference · Chat: a streaming chat console over /v1/chat/completions. Thin
   client — the full message history lives here and is replayed on each request;
   the server holds no conversation state. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "chat",
    label: "Chat",
    section: "inference",
    requires: "logits",
    async mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);

      let modelID = "overgo";
      overgo.api.get("/models").then((m) => {
        if (m && m.models && m.models[0] && m.models[0].name) modelID = m.models[0].name;
      }).catch(() => {});

      const messages = []; // {role, content}
      let controller = null;

      const system = el("textarea", { class: "text", placeholder: "system prompt (optional)…", style: "min-height:52px" });
      const log = el("div", { class: "chat-log" });
      const input = el("textarea", { class: "text", placeholder: "message…  (Enter to send, Shift+Enter for newline)" });
      const temperature = el("input", { class: "keyfield", type: "number", value: "0.7", step: "0.1", min: "0", max: "2", style: "width:80px" });
      const maxTokens = el("input", { class: "keyfield", type: "number", value: "512", min: "1", max: "4096", style: "width:90px" });
      const send = el("button", { class: "btn", onclick: submit }, "send");
      const stop = el("button", { class: "btn alt", onclick: abort, style: "display:none" }, "stop");
      const reset = el("button", { class: "btn alt", onclick: clearChat }, "clear");

      panel.append(
        el("details", { style: "margin-bottom:10px" }, el("summary", { class: "note" }, "system prompt"), system),
        log,
        input,
        el("div", { class: "chat-controls" },
          send, stop, reset,
          el("span", { class: "note", text: "temp" }), temperature,
          el("span", { class: "note", text: "max tokens" }), maxTokens));

      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(); }
      });

      function messageNode(message, streaming) {
        const body = el("div", { class: "body" });
        // Completed assistant turns render as markdown (safe DOM, no innerHTML);
        // the in-flight turn and user text stay plain so streaming stays cheap
        // and partial code fences don't flicker.
        if (message.role === "assistant" && !streaming && message.content) {
          body.appendChild(overgo.md(message.content));
        } else {
          body.appendChild(document.createTextNode(message.content));
          if (streaming) body.appendChild(el("span", { class: "cursor", text: "▋" }));
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

      function clearChat() {
        messages.length = 0;
        renderLog(false);
      }

      function abort() {
        if (controller) controller.abort();
      }

      function authHeaders(base) {
        const headers = Object.assign({}, base);
        const key = overgo.getKey();
        if (key) headers["Authorization"] = "Bearer " + key;
        return headers;
      }

      async function submit() {
        const text = input.value.trim();
        if (!text || controller) return;
        input.value = "";
        messages.push({ role: "user", content: text });
        const assistant = { role: "assistant", content: "" };
        messages.push(assistant);
        renderLog(true);

        const payload = [];
        if (system.value.trim()) payload.push({ role: "system", content: system.value.trim() });
        for (const message of messages.slice(0, -1)) payload.push({ role: message.role, content: message.content });

        controller = new AbortController();
        send.disabled = true;
        stop.style.display = "";
        try {
          const response = await fetch("/v1/chat/completions", {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({
              model: modelID,
              messages: payload,
              stream: true,
              temperature: Number(temperature.value),
              max_tokens: Number(maxTokens.value) || 512,
            }),
            signal: controller.signal,
          });
          if (!response.ok) {
            const body = await response.text();
            throw new Error(response.status === 401 ? "API key required — enter it in the top bar." : (body || ("HTTP " + response.status)));
          }
          await consume(response, assistant);
        } catch (err) {
          if (err.name === "AbortError") {
            assistant.content += (assistant.content ? "\n" : "") + "[stopped]";
          } else {
            assistant.content = "⚠ " + String(err.message || err);
          }
        } finally {
          controller = null;
          send.disabled = false;
          stop.style.display = "none";
          renderLog(false);
        }
      }

      // consume: parse the OpenAI-style SSE stream, appending each delta live.
      async function consume(response, assistant) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;
            const data = trimmed.slice(5).trim();
            if (data === "[DONE]") return;
            let parsed;
            try { parsed = JSON.parse(data); } catch (_) { continue; }
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
