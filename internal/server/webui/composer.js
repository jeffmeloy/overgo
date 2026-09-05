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

  // artifactOf: the artifact identity an artifact content URL names, for provenance.
  function artifactOf(url) { return new URL(url, location.origin).searchParams.get("id") || ""; }

  // media: a generation result whose data[] carries artifact URLs.
  async function* media(kind, result, caption) {
    for (const item of (result && result.data) || []) {
      yield { type: "media", kind, url: item.url, caption, artifact: artifactOf(item.url) };
    }
    yield { type: "done" };
  }

  // blob: a raw media body (speech synthesis) as one media event, with the artifact it was stored as.
  async function* blob(kind, body, caption, artifact) {
    yield { type: "media", kind, url: URL.createObjectURL(body), caption, bytes: body.size, mime: body.type, artifact };
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
        el("div", { class: "note" }, [event.caption, facts.join(" · ")].filter(Boolean).join(" — "),
          event.artifact ? el("span", {}, " — stored as ", overgo.artifactLink(event.artifact)) : null));
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

    // consume drives one turn from an event source: tokens land in the message (markdown at done),
    // tool and media events become cards, usage is returned as the terminal facts.
    async function consume(events, message) {
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

  function mediaKind(mime) {
    return mime.startsWith("image/") ? "image" : mime.startsWith("audio/") ? "audio" : mime.startsWith("video/") ? "video" : "document";
  }

  // ---- composer: the input surface ----
  // One prompt box (Enter sends, Shift+Enter newline), attach, drop or paste
  // the kinds the capability document accepts, an attachment strip with
  // previews and refusals, send and stop, and optional modes. Attachments
  // become the served protocol's own content parts; there is no side channel.
  function composer(host, options) {
    options = options || {};
    const input = el("textarea", { class: "text", placeholder: options.placeholder || "message (Enter to send, Shift+Enter for newline)" });
    const attachments = [];
    const attachmentHost = el("div", { class: "row" });
    // Accepted media and its bounds come from the capability document, never
    // from a list typed into a surface; a surface may narrow it to kinds.
    const media = (overgo.capabilities() || {}).media || { accept: [] };
    const accept = (media.accept || []).filter((mime) => !options.kinds || options.kinds.includes(mediaKind(mime)) || (options.kinds.includes("video") && mime === "image/gif"));
    const picker = el("input", { type: "file", style: "display:none", multiple: options.multiple !== false, accept: accept.join(",") });
    const send = el("button", { class: "btn" }, options.sendLabel || "send");
    const stop = el("button", { class: "btn alt", style: "display:none" }, "stop");
    const attach = accept.length ? el("button", { class: "btn alt", onclick: () => picker.click() }, options.attachLabel || "attach") : null;
    const modeSelect = options.modes && options.modes.length > 1
      ? el("select", { class: "text", style: "width:auto", "aria-label": "mode" }, ...options.modes.map((mode) => el("option", { value: mode.id, text: mode.label })))
      : null;
    const controls = el("div", { class: "chat-controls" }, send, stop, attach, picker, modeSelect, ...(options.controls || []));
    const element = el("div", { class: "composer" }, input, attachmentHost, controls);
    host.appendChild(element);

    function renderAttachments() {
      attachmentHost.replaceChildren(...attachments.map((item, index) => {
        const remove = el("button", { class: "btn alt", text: "×" });
        remove.addEventListener("click", () => { attachments.splice(index, 1); renderAttachments(); });
        const preview = item.refusal ? el("span", { class: "tag control", text: "refused" })
          : item.kind === "image" ? el("img", { src: item.dataURL, style: "max-height:48px;max-width:96px" })
            : item.kind === "video" ? el("video", { src: item.dataURL, style: "max-height:48px;max-width:96px" })
              : item.kind === "audio" ? el("audio", { src: item.dataURL, controls: "" })
                : el("span", { class: "tag", text: item.kind + " · " + overgo.fmt.bytes(item.size) });
        return el("span", { class: "card" + (item.refusal ? " refused" : "") }, preview, " " + item.name + " ",
          item.refusal ? el("span", { class: "note", text: item.refusal }) : null, remove);
      }));
    }
    function refusal(file, kind) {
      const refusals = media.refusals || {};
      if (!accept.includes(file.type)) return refusals[file.type] || refusals[kind] || "the served model does not accept " + (file.type || "this file");
      const limit = kind === "image" ? media.max_image_bytes : media.max_media_bytes;
      return limit && file.size > limit ? "exceeds the " + overgo.fmt.bytes(limit) + " limit" : "";
    }
    function addFile(file) {
      const kind = mediaKind(file.type);
      const item = { kind, name: file.name, mime: file.type, size: file.size, refusal: refusal(file, kind) };
      attachments.push(item);
      const reader = new FileReader();
      reader.onload = () => {
        item.dataURL = reader.result;
        if (kind === "image" && !item.refusal) {
          const probe = new Image();
          probe.onload = () => {
            if (probe.width > media.max_image_dimension || probe.height > media.max_image_dimension) item.refusal = "exceeds " + media.max_image_dimension + " pixels on a side";
            else if (probe.width * probe.height > media.max_image_pixels) item.refusal = "exceeds " + media.max_image_pixels + " pixels";
            renderAttachments();
          };
          probe.src = reader.result;
        }
        renderAttachments();
      };
      reader.readAsDataURL(file);
    }
    function addFiles(files) { for (const file of files || []) addFile(file); }
    picker.addEventListener("change", () => { addFiles(picker.files); picker.value = ""; });
    // Files also arrive by drop anywhere on the composer and by paste into the prompt.
    element.addEventListener("dragover", (event) => { event.preventDefault(); element.classList.add("drop"); });
    element.addEventListener("dragleave", () => element.classList.remove("drop"));
    element.addEventListener("drop", (event) => { event.preventDefault(); element.classList.remove("drop"); addFiles(event.dataTransfer.files); });
    input.addEventListener("paste", (event) => { if (event.clipboardData.files.length) { event.preventDefault(); addFiles(event.clipboardData.files); } });
    function attachmentParts() {
      return attachments.filter((item) => item.dataURL && !item.refusal).map((item) => {
        if (item.kind === "image") return { type: "image_url", image_url: { url: item.dataURL } };
        if (item.kind === "audio") return { type: "input_audio", input_audio: { data: item.dataURL.split(",").pop(), format: "wav" } };
        if (item.kind === "video") return { type: "input_video", input_video: { data: item.dataURL } };
        return { type: "input_file", filename: item.name, file_data: item.dataURL };
      });
    }
    function setBusy(busy) {
      send.disabled = busy;
      stop.style.display = busy ? "" : "none";
    }
    async function submit() {
      const text = input.value.trim();
      if (!text && !attachments.length) return;
      if (send.disabled || attachments.some((item) => item.refusal)) return;
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

  // userLine: the user's turn as the thread shows it, attachments named.
  function userLine(text, attachments) {
    return attachments.length ? text + "\n[" + attachments.map((item) => item.kind + ": " + item.name).join(", ") + "]" : text;
  }

  // generate: one request per composer mode beyond chat, answered in the event
  // vocabulary; the front page's modes and the generation tabs share it.
  async function* generate(mode, text, parts, signal) {
    const api = overgo.api;
    if (mode === "image-gen") { yield* media("image", await api.post("/v1/images/generations", { prompt: text }, { signal }), text); return; }
    if (mode === "video-gen") { yield* media("video", await api.post("/v1/videos/generations", { prompt: text }, { signal }), text); return; }
    if (mode === "video-edit") {
      const source = parts.find((part) => part.type === "input_video");
      if (!source) { yield { type: "error", message: "attach the source video first" }; return; }
      yield* media("video", await api.post("/v1/videos/edits", { prompt: text, source: source.input_video.data }, { signal }), text);
      return;
    }
    if (mode === "speech") {
      const response = await api.stream("/v1/audio/speech", { input: text }, { signal });
      yield* blob("audio", await response.blob(), text, response.headers.get("X-Overgo-Artifact") || "");
      return;
    }
    if (mode === "embeddings") {
      const result = await api.post("/v1/embeddings", { input: text }, { signal });
      const vector = ((result.data || [])[0] || {}).embedding || [];
      const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
      yield { type: "token", text: "embedding · " + vector.length + " dimensions · norm " + norm.toFixed(4) + "\n\n`[" + vector.slice(0, 8).map((value) => Number(value).toFixed(4)).join(", ") + (vector.length > 8 ? ", …" : "") + "]`" };
      yield { type: "usage", usage: result.usage || null, timings: null };
      yield { type: "done" };
      return;
    }
    if (mode === "rerank") {
      const [query, ...documents] = text.split("\n").map((line) => line.trim()).filter(Boolean);
      const result = await api.post("/v1/rerank", { query, documents, return_text: true }, { signal });
      const rows = (result.results || []).map((item, rank) => "| " + (rank + 1) + " | " + Number(item.relevance_score != null ? item.relevance_score : item.score).toFixed(4) + " | " + (item.text || documents[item.index] || "") + " |");
      yield { type: "token", text: "| rank | score | document |\n|---|---|---|\n" + rows.join("\n") };
      yield { type: "usage", usage: result.usage || null, timings: null };
      yield { type: "done" };
      return;
    }
    yield { type: "error", message: "the served model declares no mode " + mode };
  }

  // generationTab: a workbench tab that is one composer mode over its own thread.
  function generationTab(id, mode, options) {
    overgo.registerTab({
      id,
      mount(panel) {
        clear(panel);
        let controller = null;
        const thread = overgo.thread(panel);
        const composer = overgo.composer(panel, Object.assign({
          onStop: () => { if (controller) controller.abort(); },
          onSubmit: async (text, attachments) => {
            if (controller) return;
            controller = new AbortController();
            composer.setBusy(true);
            thread.add("user", userLine(text, attachments));
            const parts = composer.attachmentParts();
            composer.clearInput();
            composer.clearAttachments();
            try {
              await thread.consume(generate(mode, text, parts, controller.signal));
            } catch (err) {
              thread.errorRow(err.name === "AbortError" ? "cancelled" : overgo.friendlyError(err));
            } finally { controller = null; composer.setBusy(false); }
          },
        }, options));
      },
    });
  }

  overgo.thread = thread;
  overgo.composer = composer;
  overgo.streams = { reply, media, blob, responses };
  overgo.userLine = userLine;
  overgo.generate = generate;
  overgo.generationTab = generationTab;
})();
