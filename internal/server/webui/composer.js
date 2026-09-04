/* composer.js: one composer and one stream renderer for every surface that
   asks the served model for something; the renderer consumes the server's
   event vocabulary (stream_events.go) through the adapters below. */
(function () {
  "use strict";
  const overgo = window.overgo;
  const { el, clear, fmt } = overgo;

  // ---- adapters: served protocols to the event vocabulary ----

  // reply: a complete (non-streamed) chat response, rendered as one token.
  async function* reply(result) {
    const answer = result && result.choices && result.choices[0] && result.choices[0].message;
    if (!answer || typeof answer.content !== "string") {
      yield { type: "error", message: "response omitted visible content" };
      return;
    }
    yield { type: "token", text: answer.content };
    if (result.usage) yield { type: "usage", usage: result.usage, timings: result.timings || null };
    yield { type: "done" };
  }

  // media: a generation result whose data[] carries artifact URLs.
  async function* media(kind, result, caption) {
    for (const item of (result && result.data) || []) {
      yield { type: "media", kind, url: item.url, caption };
    }
    yield { type: "done" };
  }

  // blob: a raw media body (speech synthesis) as one media event.
  async function* blob(kind, body, caption) {
    yield { type: "media", kind, url: URL.createObjectURL(body), caption, bytes: body.size, mime: body.type };
    yield { type: "done" };
  }

  // responses: the Responses API named SSE events (what /interactions/follow replays too): deltas
  // as tokens, function-call items as tool events, usage+timings at completion, failures as errors.
  async function* responses(response) {
    let id = "";
    for await (const { event, data: parsed } of overgo.sseEvents(response)) {
      if (!parsed) continue;
      if (parsed.response && parsed.response.id) id = parsed.response.id;
      if (event === "response.created") yield { type: "created", id };
      else if (event === "response.output_text.delta") yield { type: "token", text: parsed.delta || "" };
      else if (event === "response.output_item.added" && parsed.item && parsed.item.type === "function_call") yield { type: "tool_start", id: parsed.item.id, name: parsed.item.name, arguments: parsed.item.arguments };
      else if (event === "response.output_item.done" && parsed.item && parsed.item.type === "function_call") yield { type: "tool_end", id: parsed.item.id, name: parsed.item.name, result: parsed.item.arguments };
      else if (event === "response.failed") yield { type: "error", message: String(parsed.delta || (parsed.error && parsed.error.message) || "the turn failed") };
      else if (event === "response.completed") {
        const usage = parsed.response && parsed.response.usage;
        yield { type: "usage", usage: usage ? { prompt_tokens: usage.input_tokens, completion_tokens: usage.output_tokens } : null, timings: (parsed.response && parsed.response.timings) || null };
      }
    }
    yield { type: "done", id };
  }

  // ---- thread: the stream renderer (messages, tool cards, media cards,
  // a thinking row, error rows) ----
  function thread(host) {
    const log = el("div", { class: "chat-log" });
    host.appendChild(log);
    const messages = [];
    let thinkingRow = null;

    function scroll() { log.scrollTop = log.scrollHeight; }

    function renderMessage(message, streaming) {
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
      const node = el("div", { class: "msg " + message.role }, head, body);
      if (message.node) message.node.replaceWith(node); else log.appendChild(node);
      message.node = node;
      scroll();
    }

    function add(role, content) {
      const message = { role, content: content || "" };
      messages.push(message);
      renderMessage(message, false);
      return message;
    }

    function thinking(on) {
      if (on && !thinkingRow) {
        thinkingRow = el("div", { class: "msg thinking", text: "thinking…" });
        log.appendChild(thinkingRow);
        scroll();
      } else if (!on && thinkingRow) { thinkingRow.remove(); thinkingRow = null; }
    }

    function toolCard(call) {
      const status = el("span", { class: "tag", text: "running" });
      const arrow = el("span", { class: "arrow", text: "▸" });
      const bodyNode = el("div", { class: "tool-body", style: "display:none" },
        el("div", { class: "note", text: "input" }),
        el("pre", { class: "mono", text: JSON.stringify(call.arguments == null ? {} : call.arguments, null, 2) }));
      const header = el("div", { class: "tool-header row" }, arrow, el("span", { class: "mono", text: call.name }), status);
      header.addEventListener("click", () => {
        const open = bodyNode.style.display === "none";
        bodyNode.style.display = open ? "" : "none";
        arrow.textContent = open ? "▾" : "▸";
      });
      const card = el("div", { class: "card tool-call" }, header, bodyNode);
      log.appendChild(card);
      scroll();
      const started = Date.now();
      return {
        end(result) {
          const elapsed = ((Date.now() - started) / 1000).toFixed(1) + "s";
          const failed = !!(result && result.error);
          status.className = failed ? "tag tag-danger" : "tag";
          status.textContent = failed ? "error · " + elapsed : "done · " + elapsed;
          bodyNode.append(el("div", { class: "note", text: failed ? "error" : "result" }),
            el("pre", { class: "mono", text: typeof result === "string" ? result : JSON.stringify(result == null ? null : result, null, 2) }));
          scroll();
        },
      };
    }

    function mediaCard(event) {
      let player;
      if (event.kind === "image") player = el("a", { href: event.url, target: "_blank" }, el("img", { src: event.url, alt: event.caption || "" }));
      else if (event.kind === "video") player = el("video", { src: event.url, controls: "", style: "max-width:420px" });
      else player = el("audio", { controls: "", src: event.url });
      const facts = [];
      if (event.mime) facts.push(event.mime);
      if (event.bytes) facts.push(fmt.bytes(event.bytes));
      const card = el("div", { class: "artifact msg media" }, player,
        el("div", { class: "note", text: [event.caption, facts.join(" · ")].filter(Boolean).join(" — ") }));
      log.appendChild(card);
      scroll();
      return card;
    }

    function errorRow(message) {
      const row = el("div", { class: "msg error" }, el("div", { class: "role", text: "error" }),
        el("div", { class: "body", text: message }));
      log.appendChild(row);
      scroll();
      return row;
    }

    // consume drives one turn from an event source: assistant tokens land in
    // the given message (streamed, then rendered as markdown at done), tool
    // consume drives one turn from an event source: tokens land in the message (markdown at done),
    // tool and media events become cards, usage is returned as the terminal facts.
      const open = new Map();
      thinking(true);
      try {
        for await (const event of events) {
          switch (event.type) {
            case "token":
              thinking(false);
              if (!assistant) assistant = add("assistant", "");
              assistant.content += event.text;
              renderMessage(assistant, true);
              break;
            case "tool_start": {
              thinking(false);
              const id = event.id || event.name + ":" + open.size;
              open.set(id, toolCard(event));
              break;
            }
            case "tool_end": {
              const id = event.id || event.name + ":" + (open.size - 1);
              const card = open.get(id) || toolCard(event);
              card.end(event.error ? { error: event.error } : event.result);
              open.delete(id);
              thinking(true);
              break;
            }
            case "media":
              thinking(false);
              mediaCard(event);
              break;
            case "usage":
              terminal.usage = event.usage;
              terminal.timings = event.timings;
              break;
            case "error":
              thinking(false);
              errorRow(event.message);
              break;
            case "created":
            case "done":
              break;
          }
        }
      } finally {
        thinking(false);
        if (assistant) renderMessage(assistant, false);
      }
      return terminal;
    }

    function reset() { messages.length = 0; clear(log); thinkingRow = null; }

    return { add, consume, toolCard, mediaCard, errorRow, thinking, reset, messages, node: log, renderMessage };
  }

  // ---- composer: the input surface ----
  // One prompt box (Enter sends, Shift+Enter newline), attach with the
  // accepted media kinds the surface declares, an attachment strip with
  // previews, send and stop, and optional modes. Attachments become the
  // served protocol's own content parts; there is no side channel.
  function composer(host, options) {
    options = options || {};
    const input = el("textarea", { class: "text", placeholder: options.placeholder || "message (Enter to send, Shift+Enter for newline)" });
    const attachments = [];
    const attachmentHost = el("div", { class: "row" });
    // Accepted media comes from the capability document, never from a list
    // typed into a surface; a surface may narrow it to kinds (image, audio, video).
    const media = (overgo.capabilities() || {}).media || { accept: [] };
    const accept = (media.accept || []).filter((mime) => !options.kinds || options.kinds.some((kind) => mime.startsWith(kind + "/") || (kind === "video" && mime === "image/gif")));
    const picker = el("input", { type: "file", style: "display:none", multiple: options.multiple !== false, accept: accept.join(",") });
    const send = el("button", { class: "btn" }, options.sendLabel || "send");
    const stop = el("button", { class: "btn alt", style: "display:none" }, "stop");
    const attach = accept.length ? el("button", { class: "btn alt", onclick: () => picker.click() }, options.attachLabel || "attach") : null;
    const modeSelect = options.modes && options.modes.length
      ? el("select", { class: "text", style: "width:auto" }, ...options.modes.map((mode) => el("option", { value: mode.id, text: mode.label })))
      : null;
    const controls = el("div", { class: "chat-controls" }, send, stop, attach, picker, modeSelect, ...(options.controls || []));
    const element = el("div", { class: "composer" }, input, attachmentHost, controls);
    host.appendChild(element);

    function renderAttachments() {
      attachmentHost.replaceChildren(...attachments.map((item, index) => {
        const remove = el("button", { class: "btn alt", text: "×" });
        remove.addEventListener("click", () => { attachments.splice(index, 1); renderAttachments(); });
        const preview = item.kind === "image"
          ? el("img", { src: item.dataURL, style: "max-height:48px;max-width:96px" })
          : item.kind === "video"
            ? el("video", { src: item.dataURL, style: "max-height:48px;max-width:96px" })
            : el("span", { class: "tag", text: item.kind });
        return el("span", { class: "card" }, preview, " " + item.name + " ", remove);
      }));
    }
    function addFile(file) {
      const reader = new FileReader();
      reader.onload = () => {
        const kind = file.type.startsWith("image/") ? "image"
          : file.type.startsWith("audio/") ? "audio"
            : file.type.startsWith("video/") ? "video" : "document";
        attachments.push({ kind, name: file.name, mime: file.type, dataURL: reader.result });
        renderAttachments();
      };
      reader.readAsDataURL(file);
    }
    picker.addEventListener("change", () => {
      for (const file of picker.files) addFile(file);
      picker.value = "";
    });
    function attachmentParts() {
      return attachments.map((item) => {
        if (item.kind === "image") return { type: "image_url", image_url: { url: item.dataURL } };
        if (item.kind === "audio") return { type: "input_audio", input_audio: { data: item.dataURL.split(",").pop(), format: "wav" } };
        if (item.kind === "video") return { type: "input_video", input_video: { data: item.dataURL } };
        return { type: "text", text: "[document " + item.name + "]" };
      });
    }
    function setBusy(busy) {
      send.disabled = busy;
      stop.style.display = busy ? "" : "none";
    }
    async function submit() {
      const text = input.value.trim();
      if (!text && !attachments.length) return;
      if (send.disabled) return;
      if (options.onSubmit) await options.onSubmit(text, attachments.slice(), modeSelect ? modeSelect.value : "");
    }
    send.addEventListener("click", submit);
    stop.addEventListener("click", () => { if (options.onStop) options.onStop(); });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(); }
    });
    return {
      element, input, attachments, attachmentParts, setBusy, addFile,
      clearAttachments() { attachments.length = 0; renderAttachments(); },
      openPicker() { picker.click(); },
      clearInput() { input.value = ""; },
      mode() { return modeSelect ? modeSelect.value : ""; },
    };
  }

  overgo.thread = thread;
  overgo.composer = composer;
  overgo.streams = { reply, media, blob, responses };
})();
