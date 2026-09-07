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

  // media: a generation result whose data[] carries artifact URLs; record names the run and
  // capability behind them so a card can replay the stored request.
  async function* media(kind, result, caption, record) {
    for (const item of (result && result.data) || []) yield { type: "media", kind, url: item.url, caption, artifact: artifactOf(item.url), ...record };
    yield { type: "done" };
  }

  // run: one request of a declared capability through the generic run route, answered as the
  // operation's outputs (artifact URLs) once it completes, or thrown as its failure.
  async function run(capability, input, signal, sources) {
    const accepted = await overgo.api.post("/generation/run", { task: capability.task, recipe: capability.recipe, input, sources: sources || [] }, { signal });
    const completed = await overgo.waitOperation(accepted.operation, null, signal);
    if (completed.state !== "completed") throw new Error(completed.failure || completed.state);
    return { run: completed.run, data: (completed.outputs || []).map((id) => ({ url: "/artifacts/content?id=" + encodeURIComponent(id) })) };
  }

  // replay: the stored request of a record resubmitted (unchanged, or varied by the page) as a new
  // run; each output names its parent, and one identical to the parent says the store memoized it.
  async function* replay(capability, input, signal, parent, label) {
    const completed = await run(capability, input, signal);
    for await (const event of media(outputKind(capability.task), completed, label + " of " + fmt.shortID(parent), { run: completed.run, recipe: capability.recipe })) {
      if (event.type === "media" && event.artifact === parent) event.caption += " — the same output: the store memoized the unchanged request";
      yield event;
    }
  }

  // generation: one run of a declared capability through the generic run route:
  // the operation's outputs land as media events, each an artifact with provenance,
  // or as the assistant's text when the task answers in text; the output's kind
  // follows the server's task vocabulary, not a model list.
  const outputKind = (task) => task === "speech" ? "audio" : task === "vqa" || task === "transcription" ? "text" : task.startsWith("video") ? "video" : "image";
  // bodyControl: the declared text control the message body feeds (a prompt, a text, a question).
  const bodyControl = (controls) => (controls || []).find((control) => control.type === "text" && ["prompt", "text", "question"].includes(control.name));
  async function* generation(selection, text, signal) {
    const { capability, fields } = selection;
    const { input, missing } = overgo.controlValues(fields);
    const textControl = bodyControl(capability.controls);
    if (textControl) { input[textControl.name] = text; missing.delete(textControl.name); }
    if (missing.size) { yield { type: "error", message: [...missing].join(", ") + " required" }; return; }
    let completed;
    try { completed = await run(capability, input, signal, selection.sources); } catch (err) { if (err.name === "AbortError") throw err; yield { type: "error", message: err.message }; return; }
    if (outputKind(capability.task) !== "text") { yield* media(outputKind(capability.task), completed, text, { run: completed.run, recipe: capability.recipe }); return; }
    // Text outputs only: a run's document outputs (a transcription record) stay stored beside them.
    for (const output of completed.data) { const blob = await overgo.api.blob(output.url); if (blob.type.startsWith("text/")) yield { type: "token", text: await blob.text() }; }
    yield { type: "done" };
  }

  // responses: the Responses API SSE events (follow replays them too): deltas, function-call items, usage, failures.
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

  // ---- thread: the stream renderer (messages, tool cards, media cards, a thinking row, error rows);
  // options.reuse(file): media output -> next turn's input; options.marker: tag on every assistant turn ("remote"). ----
  function thread(host, options) {
    const reuse = options && options.reuse;
    const replay = options && options.replay; // replay(event, "regenerate" | "vary"): the page resubmits the card's stored request
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
      if (message.role === "assistant" && options && options.marker) head.appendChild(el("span", { class: "tag", text: options.marker }));
      if (message.role === "assistant" && !streaming && message.content) {
        head.appendChild(overgo.copyButton(message.content, "copy"));
        if (message.response && overgo.inspectTurn) head.appendChild(el("button", { class: "link-button", text: "inspect", onclick: () => overgo.inspectTurn(message.response) }));
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
      const player = mediaPlayer(event.kind, event.url, event.caption);
      const facts = [];
      if (event.mime) facts.push(event.mime);
      if (event.bytes) facts.push(fmt.bytes(event.bytes));
      // A media output offers itself back as input: its bytes come from the
      // artifact the store holds and re-enter the composer as its own kind.
      const again = reuse && event.url ? el("button", { class: "btn alt", text: "use as input", onclick: async () => {
        try {
          const body = await overgo.api.blob(event.url);
          reuse(new File([body], (event.artifact || "output").replace(/[^A-Za-z0-9]+/g, "-").slice(0, 40) + "." + (body.type.split("/").pop() || "bin"), { type: body.type }), event.artifact);
        } catch (err) { errorRow(overgo.friendlyError(err)); }
      } }) : null;
      // A card behind a run replays its stored request: unchanged, or with a fresh seed.
      const replays = replay && event.run ? ["regenerate", "vary"].map((label) => el("button", { class: "btn alt", text: label, onclick: () => replay(event, label) })) : [];
      const lineage = options && options.lineage && event.artifact ? el("button", { class: "btn alt", text: "lineage", onclick: () => options.lineage(event, card) }) : null;
      const card = el("div", { class: "artifact msg media" }, player,
        el("div", { class: "note" }, [event.caption, facts.join(" · ")].filter(Boolean).join(" — "),
          event.artifact ? el("span", {}, " — stored as ", overgo.artifactLink(event.artifact)) : null, again, ...replays, lineage));
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
    async function consume(events, assistant) {
      const terminal = {};
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
            case "tool_start": { thinking(false); const id = event.id || event.name + ":" + open.size; open.set(id, toolCard(event)); break; }
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
      } finally { thinking(false); if (assistant) renderMessage(assistant, false); }
      return terminal;
    }

    function reset() { messages.length = 0; clear(log); thinkingRow = null; }

    return { add, consume, toolCard, mediaCard, errorRow, thinking, reset, messages, node: log, renderMessage };
  }

  // mediaPlayer: the element that shows a media artifact as what it is (image, video, audio).
  function mediaPlayer(kind, url, caption) {
    if (kind === "image") return el("a", { href: url, target: "_blank" }, el("img", { src: url, alt: caption || "" }));
    if (kind === "video") return el("video", { src: url, controls: "", style: "max-width:420px" });
    return el("audio", { controls: "", src: url });
  }

  function mediaKind(mime) { return mime.startsWith("image/") ? "image" : mime.startsWith("audio/") ? "audio" : mime.startsWith("video/") ? "video" : "document"; }

  // ---- composer: the input surface: one prompt box (Enter sends, Shift+Enter newline), attach, drop or
  // paste the kinds the capability document accepts, an attachment strip with previews and refusals, send
  // and stop, optional modes; attachments become the served protocol's own content parts. ----
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
    // openPicker: the dialog filters to the served model's types unless the surface takes any file (options.takesAny).
    function openPicker() { picker.accept = options.takesAny && options.takesAny() ? "" : accept.join(","); picker.click(); }
    const attach = accept.length ? el("button", { class: "btn alt", onclick: openPicker }, options.attachLabel || "attach") : null;
    const modeSelect = options.modes && options.modes.length > 1 ? el("select", { class: "text", style: "width:auto", "aria-label": "mode" }, ...options.modes.map((mode) => el("option", { value: mode.id, text: mode.label }))) : null;
    // modeHost: what a generation mode declares (its model, its controls) rendered by the page.
    const modeHost = el("span", { class: "row mode-controls" });
    const controls = el("div", { class: "chat-controls" }, send, stop, attach, picker, modeSelect, modeHost, ...(options.controls || []));
    const element = el("div", { class: "composer" }, input, attachmentHost, controls);
    host.appendChild(element);

    function renderAttachments() {
      attachmentHost.replaceChildren(...attachments.map((item, index) => {
        const remove = el("button", { class: "btn alt", text: "×", onclick: () => { attachments.splice(index, 1); renderAttachments(); } });
        const preview = item.refusal ? el("span", { class: "tag control", text: "refused" }) : item.kind === "image" ? el("img", { src: item.dataURL, style: "max-height:48px;max-width:96px" })
          : item.kind === "video" ? el("video", { src: item.dataURL, style: "max-height:48px;max-width:96px" }) : item.kind === "audio" ? el("audio", { src: item.dataURL, controls: "" }) : el("span", { class: "tag", text: item.kind + " · " + overgo.fmt.bytes(item.size) });
        return el("span", { class: "card" + (item.refusal ? " refused" : "") }, preview, " " + item.name + " ",
          item.refusal ? el("span", { class: "note", text: item.refusal }) : item.artifact ? el("span", { class: "note" }, "stored as ", overgo.artifactLink(item.artifact)) : null, remove);
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
      // options.intake(file): a surface storing the file as an artifact returns the id's promise; the server's refusal is the card's, and the card sends no part.
      const stored = options.intake ? options.intake(file) : null;
      const item = { kind, name: file.name, mime: file.type, size: file.size, refusal: stored ? "" : refusal(file, kind), storing: !!stored };
      if (stored) stored.then((id) => { item.artifact = id; }, (err) => { item.refusal = overgo.friendlyError(err); }).then(() => { item.storing = false; renderAttachments(); });
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
      return attachments.filter((item) => item.dataURL && !item.refusal && !item.artifact && !item.storing).map((item) => {
        if (item.kind === "image") return { type: "image_url", image_url: { url: item.dataURL } };
        if (item.kind === "audio") return { type: "input_audio", input_audio: { data: item.dataURL.split(",").pop(), format: "wav" } };
        if (item.kind === "video") return { type: "input_video", input_video: { data: item.dataURL } };
        return { type: "input_file", filename: item.name, file_data: item.dataURL };
      });
    }
    function setBusy(busy) { send.disabled = busy; stop.style.display = busy ? "" : "none"; }
    async function submit() {
      const text = input.value.trim();
      if (!text && !attachments.length) return;
      if (send.disabled || attachments.some((item) => item.refusal || item.storing)) return;
      if (options.onSubmit) await options.onSubmit(text, attachments.slice(), modeSelect ? modeSelect.value : "");
    }
    send.addEventListener("click", submit);
    stop.addEventListener("click", () => { if (options.onStop) options.onStop(); });
    if (modeSelect && options.onMode) modeSelect.addEventListener("change", () => options.onMode(modeSelect.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(); }
    });
    return {
      element, input, attachments, attachmentParts, setBusy, addFile, modeHost,
      clearAttachments() { attachments.length = 0; renderAttachments(); },
      openPicker,
      clearInput() { input.value = ""; },
      mode() { return modeSelect ? modeSelect.value : ""; },
      setMode(id) { if (!modeSelect) return null; modeSelect.value = id; return options.onMode ? options.onMode(id) : null; },
    };
  }

  // userLine: the user's turn as the thread shows it, attachments named.
  function userLine(text, attachments) { return attachments.length ? text + "\n[" + attachments.map((item) => item.kind + ": " + item.name).join(", ") + "]" : text; }

  // generate: one request per composer mode beyond chat, answered in the event
  // vocabulary; the front page's modes and the generation tabs share it. A
  // generation mode carries the declared capability the page selected and the
  // controls it typed; nothing about a task's request lives here.
  async function* generate(mode, text, parts, signal, selection) {
    const api = overgo.api;
    if (selection && selection.capability) { yield* generation(selection, text, signal); return; }
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

  // toolStep: the manual tool-step surface the agent tab and the front page share: pick an allowed tool and
  // its exact arguments, review what a grant binds (manual identity, argument bytes, any committed decision),
  // execute an inspection or approve a mutation bound to the previewed operation identity, watch steps against
  // the bound; results are the thread's tool cards and a refusal re-renders the decision.
  function toolStep(host, options) {
    const api = overgo.api;
    const select = el("select", { class: "text", "aria-label": "tool" });
    const args = el("textarea", { class: "text", rows: "2", placeholder: "Strict JSON arguments" });
    const decisionHost = el("div");
    const guard = el("span", { class: "note", "aria-label": "guardrails" });
    const review = el("button", { class: "btn alt", text: "Review decision" });
    const execute = el("button", { class: "btn alt", text: "Execute inspection" });
    const approve = el("button", { class: "btn", text: "Approve and execute", disabled: true });
    let previewed = null;
    function setAgent(agent, tools) {
      const allowed = new Set((agent && agent.tools) || []);
      select.replaceChildren(...tools.filter((item) => allowed.has(item.manual)).map((tool) => el("option", { value: tool.name, text: tool.name + " / " + tool.effect })));
      review.disabled = execute.disabled = !agent;
    }
    function body() { return { agent: options.agent(), session: options.session(), tool: select.value, arguments: JSON.parse(args.value || "{}") }; }
    function renderDecision(preview) {
      previewed = preview;
      const rows = [
        el("div", {}, el("span", { class: preview.effect === "mutation" ? "tag tag-danger" : "tag", text: preview.effect }), " ", preview.tool, " / manual ", overgo.artifactLink(preview.manual)),
        el("div", { class: "mono", text: "grant binds: " + preview.arguments.join(" ") + " / operation " + preview.operation }),
      ];
      if (!preview.decision) rows.push(el("div", { class: "note", text: preview.effect === "mutation" ? "No committed decision yet: approving records a durable grant for exactly these bytes." : "Inspections are effect-free and need no decision." }));
      else if (preview.binds) rows.push(el("div", {}, el("span", { class: "tag", text: preview.decision.answer }), " committed decision ", overgo.artifactLink(preview.decision.id), " binds these exact facts"));
      else rows.push(el("div", {}, el("span", { class: "tag tag-danger", text: "does not bind" }), " committed decision ", overgo.artifactLink(preview.decision.id), " (" + preview.decision.answer + ")"),
        el("div", { class: "mono", text: "approved: " + preview.decision.arguments.join(" ") }), el("div", { class: "mono", text: "proposed: " + preview.arguments.join(" ") }));
      decisionHost.replaceChildren(el("div", { class: "card approval" }, ...rows));
      approve.disabled = preview.effect !== "mutation";
      execute.disabled = preview.effect === "mutation";
    }
    async function preview() { renderDecision(await api.post("/agents/approval", body())); }
    async function step(approving) {
      const request = body();
      if (approving) request.approval = previewed && previewed.operation;
      const card = options.thread().toolCard({ name: request.tool, arguments: request.arguments });
      try {
        const result = await api.post("/agents/step", request);
        card.end(result.result);
        guard.textContent = "steps " + result.steps + " / " + result.bound + " · " + (result.bound - result.steps) + " remaining";
        decisionHost.replaceChildren();
        previewed = null;
        if (options.onStep) options.onStep(result);
      } catch (err) {
        card.end({ error: overgo.friendlyError(err) });
        try { await preview(); } catch (_) { /* the refusal stands on its own */ }
      }
    }
    review.addEventListener("click", () => preview().catch(options.onError));
    execute.addEventListener("click", () => step(false));
    approve.addEventListener("click", () => step(true));
    host.append(el("div", { class: "row" }, select, review, execute, approve, guard, ...(options.controls || [])), args, decisionHost);
    return { setAgent, guard };
  }


  overgo.thread = thread;
  overgo.composer = composer;
  overgo.streams = { reply, media, responses, replay };
  overgo.userLine = userLine;
  overgo.generate = generate;
  overgo.bodyControl = bodyControl;
  overgo.outputKind = outputKind;
  overgo.toolStep = toolStep;
  overgo.mediaPlayer = mediaPlayer;
  overgo.mediaKind = mediaKind;
})();
