/* composer.js: one composer and one stream renderer for every surface that
   asks the served model for something; the renderer consumes the server's
   event vocabulary (stream_events.go) through the adapters below. */
(function () {
  "use strict";
  const overgo = window.overgo;
  const { el, clear, fmt } = overgo;
  const threadScrollers = new WeakMap();
  const resizeThreads = () => document.querySelectorAll('.chat-log').forEach(log => { const scroll = threadScrollers.get(log); if (scroll) scroll(); });
  window.addEventListener("resize", resizeThreads);
  if (window.visualViewport) window.visualViewport.addEventListener("resize", resizeThreads);

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
  async function run(capability, input, signal, sources, lifecycle) {
    let accepted;
    try { accepted = await overgo.api.post("/generation/run", { task: capability.task, recipe: capability.recipe, input, sources: sources || [] }, { signal }); }
    catch (err) { err.operationUnconfirmed = !err.status && err.name !== 'AbortError'; throw err; }
    if (!accepted || typeof accepted.operation !== 'string' || !accepted.operation) { const err = new Error('The operation was submitted without a receipt. Check Activity before trying again.'); err.operationUnconfirmed = true; throw err; }
    if (lifecycle && lifecycle.accepted) await lifecycle.accepted(accepted.operation);
    const completed = await overgo.waitOperation(accepted.operation, lifecycle && lifecycle.observe, signal);
    if (completed.state !== "completed") { const err = new Error(completed.failure || completed.state); err.operationState = completed.state; throw err; }
    return { run: completed.run, data: (completed.outputs || []).map((id) => ({ url: "/artifacts/content?id=" + encodeURIComponent(id) })) };
  }

  // replay: the stored request of a record resubmitted (unchanged, or varied by the page) as a new
  // run; each output names its parent, and one identical to the parent says the store memoized it.
  async function* replay(capability, input, signal, parent, label, lifecycle) {
    let completed;
    try { completed = await run(capability, input, signal, null, lifecycle); }
    catch (err) { if (err.operationState === 'cancelled') { yield { type: 'cancelled' }; return; } throw err; }
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
    try { completed = await run(capability, input, signal, selection.sources, selection.lifecycle); } catch (err) {
      if (err.name === "AbortError") throw err;
      if (err.operationState === 'cancelled') yield { type: 'cancelled' };
      else yield { type: "error", message: err.message, status: err.operationUnconfirmed ? 'unknown' : 'failed' };
      return;
    }
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
      else if (parsed.response_id) id = parsed.response_id;
      if (event === "response.created") yield { type: "created", id };
      else if (event === "response.output_text.delta") yield { type: "token", text: parsed.delta || "" };
      else if (event === "response.output_item.added" && parsed.item && parsed.item.type === "function_call") yield { type: "tool_start", id: parsed.item.id, name: parsed.item.name, arguments: parsed.item.arguments };
      else if (event === "response.output_item.done" && parsed.item && parsed.item.type === "function_call") yield { type: "tool_end", id: parsed.item.id, name: parsed.item.name, result: parsed.item.arguments };
      else if (event === "response.failed") {
        yield { type: "error", id, status: "failed", message: String(parsed.delta || (parsed.error && parsed.error.message) || "The turn failed.") };
        return;
      } else if (event === "response.cancelled") {
        yield { type: "cancelled", id, status: "cancelled" };
        return;
      }
      else if (event === "response.completed") {
        const usage = parsed.response && parsed.response.usage;
        yield { type: "usage", usage: usage ? { prompt_tokens: usage.input_tokens, completion_tokens: usage.output_tokens } : null, timings: (parsed.response && parsed.response.timings) || null };
        yield { type: "done", id, status: "completed" };
        return;
      }
    }
    throw new Error("The connection ended before the response was confirmed. Resume to recover its result.");
  }

  // ---- thread: the stream renderer (messages, tool cards, media cards, a thinking row, error rows);
  // options.reuse(file): media output -> next turn's input; options.marker: tag on every assistant turn ("remote"). ----
  function thread(host, options) {
    const reuse = options && options.reuse;
    const replay = options && options.replay; // replay(event, "regenerate" | "vary"): the page resubmits the card's stored request
    const log = el("div", { class: "chat-log", role: "log", "aria-label": "Conversation", "aria-live": "off", tabindex: "0" });
    const announcement = el("div", { class: "sr-only", role: "status", "aria-atomic": "true" });
    const latest = el("button", { class: "link-button jump-latest", text: "Jump to latest", hidden: true, onclick: () => { following = true; scroll(); log.focus({ preventScroll: true }); } });
    host.append(log, latest, announcement);
    const messages = [];
    let thinkingRow = null;

    let following = true;
    log.addEventListener("scroll", () => {
      following = Math.ceil(log.scrollTop + log.clientHeight) >= log.scrollHeight;
      latest.hidden = following;
    });
    function scroll() {
      latest.hidden = following || log.scrollHeight <= log.clientHeight;
      if (!following) return;
      log.scrollTop = log.scrollHeight;
    }
    threadScrollers.set(log, scroll);

    function renderMessage(message, streaming) {
      const position = log.scrollTop;
      if (!message.node) { message.node = el("div", { class: "msg " + message.role }, el("div", { class: "role" }), el("div", { class: "body" })); log.appendChild(message.node); }
      const body = message.node.querySelector(".body");
      if (message.role === "assistant" && !streaming && message.content) {
        body.replaceChildren(overgo.md(message.content));
        message.streamText = null;
      } else if (streaming) {
        if (!message.streamText) {
          message.streamText = document.createTextNode(message.content);
          body.replaceChildren(message.streamText, el("span", { class: "cursor", text: "|", "aria-hidden": "true" }));
        } else if (message.content.startsWith(message.streamText.data)) message.streamText.appendData(message.content.slice(message.streamText.length));
        else message.streamText.data = message.content;
      } else {
        body.replaceChildren(document.createTextNode(message.content));
        message.streamText = null;
      }
      const head = el("div", { class: "role" }, message.role);
      if (message.role === "assistant" && options && options.marker) head.appendChild(el("span", { class: "tag", text: options.marker }));
      if (!streaming && message.content) {
        head.appendChild(overgo.copyButton(message.content, "copy"));
        if (message.response && overgo.inspectTurn) head.appendChild(el("button", { class: "link-button", text: "inspect", onclick: () => overgo.inspectTurn(message.response) }));
      }
      if (!streaming && message.response && options && options.actions) head.append(...options.actions(message));
      message.node.querySelector(".role").replaceWith(head);
      if (!following) log.scrollTop = position;
      scroll();
    }

    function add(role, content) {
      if (role === "user") following = true;
      const message = { role, content: content || "" };
      messages.push(message);
      renderMessage(message, false);
      return message;
    }

    function thinking(on) {
      if (on && !thinkingRow) {
        announcement.textContent = "Waiting for a response.";
        thinkingRow = el("div", { class: "msg thinking", text: "thinking…" });
        log.appendChild(thinkingRow);
        scroll();
      } else if (!on && thinkingRow) { thinkingRow.remove(); thinkingRow = null; }
    }

    function toolCard(call) {
      const status = el("span", { class: "tag", text: "running" });
      const arrow = el("span", { class: "arrow", text: "▸" });
      const bodyNode = el("div", { class: "tool-body", hidden: true },
        el("div", { class: "note", text: "input" }),
        el("pre", { class: "mono", text: JSON.stringify(call.arguments == null ? {} : call.arguments, null, 2) }));
      bodyNode.id = "tool-" + crypto.randomUUID();
      const header = el("button", { class: "tool-header row", type: "button", "aria-expanded": "false", "aria-controls": bodyNode.id }, arrow, el("span", { class: "mono", text: call.name }), status);
      header.addEventListener("click", () => { bodyNode.hidden = !bodyNode.hidden; header.setAttribute("aria-expanded", String(!bodyNode.hidden)); arrow.textContent = bodyNode.hidden ? "▸" : "▾"; });
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
      player.addEventListener("load", scroll, true);
      player.addEventListener("loadedmetadata", scroll, true);
      const facts = [];
      if (event.mime) facts.push(event.mime);
      if (event.bytes) facts.push(fmt.bytes(event.bytes));
      // A media output offers itself back as input: its bytes come from the
      // artifact the store holds and re-enter the composer as its own kind.
      const again = reuse && event.url ? el("button", { class: "btn alt", text: "use as input", onclick: async () => {
        try {
          const body = await overgo.api.blob(event.url);
          reuse(new File([body], (event.artifact || "output").replace(/[^A-Za-z0-9]+/g, "-").slice(0, 40) + "." + (body.type.split("/").pop() || "bin"), { type: body.type }), event.artifact, event);
        } catch (err) { errorRow(overgo.friendlyError(err)); }
      } }) : null;
      // A card behind a run replays its stored request: unchanged, or with a fresh seed.
      const replays = replay && event.run ? ["regenerate", "vary"].map((label) => el("button", { class: "btn alt media-replay", text: label, onclick: () => replay(event, label) })) : [];
      const lineage = options && options.lineage && event.artifact ? el("button", { class: "btn alt", text: "lineage", onclick: () => options.lineage(event, card) }) : null;
      const card = el("div", { class: "artifact msg media" }, player,
        el("div", { class: "note" }, [event.caption, facts.join(" · ")].filter(Boolean).join(" — "),
          event.artifact ? el("span", {}, " — stored as ", overgo.artifactLink(event.artifact)) : null, again, ...replays, lineage));
      if (event.kind === 'audio') {
        card.classList.add('audio-result');
        player.setAttribute('aria-label', event.caption || 'Generated audio');
        const details = el('details', { class: 'audio-details' }, el('summary', { text: 'Details' }),
          el('div', { class: 'note', text: [event.caption, facts.join(' · ')].filter(Boolean).join(' — ') }),
          event.artifact ? overgo.artifactLink(event.artifact) : null, lineage);
        let actions = null;
        if (again || replays.length) actions = el('details', { class: 'audio-actions' }, el('summary', { text: 'More actions' }), el('div', { class: 'row' }, again, ...replays));
        card.replaceChildren(player, el('div', { class: 'row audio-output-actions' },
          event.url ? el('a', { class: 'link-button audio-download', href: event.url, download: 'audio', text: 'Download audio' }) : null,
          actions), details);
      }
      log.appendChild(card);
      scroll();
      return card;
    }

    function errorRow(message) {
      const row = el("div", { class: "msg error", role: "alert" }, el("div", { class: "role", text: "error" }),
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
              if (!assistant || !assistant.content) announcement.textContent = "Receiving a response.";
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
              terminal.status = event.status || "failed";
              announcement.textContent = "Response failed.";
              break;
            case "cancelled":
              thinking(false);
              log.appendChild(el("div", { class: "note", role: "status", text: "Stopped" }));
              terminal.status = "cancelled";
              announcement.textContent = "Response stopped.";
              break;
            case "done":
              terminal.status = event.status || "completed";
              announcement.textContent = "Response ready.";
              break;
            case "created":
              break;
          }
        }
      } finally { thinking(false); if (assistant) renderMessage(assistant, false); }
      return terminal;
    }

    function reset() { messages.length = 0; clear(log); thinkingRow = null; following = true; latest.hidden = true; announcement.textContent = ""; }

    return { add, consume, toolCard, mediaCard, errorRow, thinking, reset, messages, node: log, renderMessage };
  }

  // mediaPlayer: the element that shows a media artifact as what it is (image, video, audio).
  function mediaPlayer(kind, url, caption) {
    if (kind === "image") return el("a", { href: url, target: "_blank" }, el("img", { src: url, alt: caption || "" }));
    if (kind === "video") return el("video", { src: url, controls: "", class: "mw-420" });
    return el("audio", { controls: "", src: url });
  }

  function mediaKind(mime) { return mime.startsWith("image/") ? "image" : mime.startsWith("audio/") ? "audio" : mime.startsWith("video/") ? "video" : "document"; }

  // ---- composer: the input surface: one prompt box (Enter sends, Shift+Enter newline), attach, drop or
  // paste the kinds the capability document accepts, an attachment strip with previews and refusals, send
  // and stop, optional modes; attachments become the served protocol's own content parts. ----
  function composer(host, options) {
    options = options || {};
    const input = el("textarea", { class: "text", "aria-label": "Message", placeholder: options.placeholder || "Message…", title: "Enter to send · Shift+Enter for a new line" });
    const attachments = [];
    let busy = false;
    let readOnly = false;
    let sendBlocked = false;
    let disposed = false;
    let paused = false;
    const attachmentHost = el("div", { class: "row attachment-strip" });
    // Accepted media and its bounds come from the capability document, never
    // from a list typed into a surface; a surface may narrow it to kinds.
    const media = (overgo.capabilities() || {}).media || { accept: [] };
    const accept = (media.accept || []).filter((mime) => !options.kinds || options.kinds.includes(mediaKind(mime)) || (options.kinds.includes("video") && mime === "image/gif"));
    const picker = el("input", { type: "file", hidden: true, multiple: options.multiple !== false, accept: accept.join(",") });
    const send = el("button", { class: "btn" }, options.sendLabel || "send");
    const stop = el("button", { class: "btn alt", hidden: true }, "stop");
    // openPicker: the dialog filters to the served model's types unless the surface takes any file (options.takesAny).
    function openPicker() { if (disposed || readOnly || paused) return; picker.multiple = selectedMode() !== 'transcription' && options.multiple !== false; picker.accept = options.takesAny && options.takesAny() ? "" : accept.join(","); picker.click(); }
    const capture = overgo.mediaCapture({ media, audio: () => options.captureAudio?.(), accept: () => options.captureAccept ? options.captureAccept() : accept, files: openPicker, addFile });
    const attach = accept.length || options.takesAny ? el("button", { class: "btn alt attach-button", onclick: () => { if (!disposed && !readOnly && !paused) capture.open(); } }, options.attachLabel || "attach") : null;
    const modeSelect = options.modes && options.modes.length > 1 ? el("select", { class: "text w-auto", "aria-label": "mode" }, ...options.modes.map((mode) => el("option", { value: mode.id, text: mode.label }))) : null;
    const selectedMode = () => modeSelect ? modeSelect.value : options.modes?.[0]?.id || '';
    // modeHost: what a generation mode declares (its model, its controls) rendered by the page.
    const modeHost = el("span", { class: "row mode-controls" });
    const controls = el("div", { class: "chat-controls" }, send, stop, attach, picker, modeSelect, ...((options.controls) || []), el("span", { class: "grow" }));
    const transcriptionPicker = el('select', { class: 'text', 'aria-label': 'Transcription model' });
    const transcriptionNote = el('span', { class: 'note', role: 'status' });
    const transcriptionRetry = el('button', { class: 'link-button', text: 'Retry model list', hidden: true, onclick: () => { transcriptionLoaded = false; renderAttachments(); } });
    const transcriptionHost = el('div', { class: 'transcription-controls', hidden: true }, el('label', {}, 'Transcription model', transcriptionPicker), transcriptionNote, transcriptionRetry);
    const extras = el("div", { class: "composer-extras" }, modeHost, transcriptionHost, attachmentHost);
    const guidanceText = el('div', { class: 'note' });
    const guidance = el('details', { class: 'attachment-guidance', hidden: true }, el('summary', { text: 'Accepted files' }), guidanceText);
    extras.appendChild(guidance);
    const element = el("div", { class: "composer" }, extras, input, controls);
    send.classList.add("send-button"); stop.classList.add("stop-button");
    host.appendChild(element);

    // Stable flat file rows keep keyboard targets and playing previews intact
    // while read/upload state changes elsewhere in the list.
    const attachmentURL = item => item.dataURL || (item.kind === 'audio' && item.artifact && '/artifacts/content?id=' + encodeURIComponent(item.artifact));
    function attachmentPreview(item) {
      const url = attachmentURL(item);
      if (!url || item.pending || item.refusal) return null;
      if (item.kind === "image") return el("img", { src: url, alt: item.name, class: "thumb-preview" });
      if (item.kind === "video") return el("video", { src: url, class: "thumb-preview" });
      if (item.kind === "audio") return el("audio", { src: url, controls: "" });
      return null;
    }
    function stage(item, phase, reason = '') {
      item.phase = phase; item.pending = ['reading', 'validating', 'fetching'].includes(phase); item.storing = phase === 'uploading'; item.refusal = reason;
    }
    function pausePreview(item) { for (const player of item.preview?.querySelectorAll('audio, video') || []) player.pause(); }
    function invalidate(item) {
      pausePreview(item);
      item.attempt = null; if (item.cancel) item.cancel();
      clearTranscript(item);
    }
    function releaseFile(item) {
      if (item.release && !attachments.some(other => other !== item && other.target === item.target && other.artifact === item.artifact)) item.release();
      item.artifact = ''; item.target = item.release = null;
      if (item.infoLink) item.infoLink.remove(); item.infoLink = item.link = null;
    }
    function removeFile(item) {
      if (disposed || readOnly || paused) return;
      const index = attachments.indexOf(item);
      if (index < 0) return;
      const focused = item.row && item.row.contains(document.activeElement);
      attachments.splice(index, 1); invalidate(item);
      releaseFile(item);
      if (item.row) item.row.remove(); renderAttachments();
      if (focused) (attachments[index]?.remove || attachments[index - 1]?.remove || attach || input).focus();
    }
    function stopFile(item) {
      if (disposed || readOnly || paused) return;
      if (item.transcribing) { stopTranscription(item, 'Transcription stopped. Transcribe again or send the recording.'); return; }
      if (!item.file && item.sourceArtifact) item.needsReattach = true;
      invalidate(item); stage(item, 'cancelled', 'File preparation stopped. Retry or remove it.'); renderAttachments(); item.retry.focus();
    }
    // ---- transcription: a recorded or attached audio file transcribes through the served model's own
    // route on explicit action; the transcript is an offer to insert or replace, never a send. ----
    const transcribable = () => selectedMode() !== 'transcription' && ((overgo.capabilities() || {}).modes || []).some((mode) => mode.id === 'transcription' && mode.enabled);
    let transcriptionLoaded = false, transcriptionLoading = null, transcriptionChoices = [], transcriptionSelected = '';
    const transcriptionRecipe = () => transcriptionChoices.find(choice => choice.recipe === transcriptionPicker.value && !choice.refusal)?.recipe;
    function loadTranscriptionChoices() {
      if (transcriptionLoaded || transcriptionLoading || disposed) return;
      const controller = new AbortController(); transcriptionLoading = controller;
      transcriptionNote.textContent = 'Loading transcription models…'; transcriptionRetry.hidden = true;
      overgo.api.get('/generation/capabilities', { signal: controller.signal }).then(choices => {
        if (disposed || transcriptionLoading !== controller) return;
        if (!Array.isArray(choices)) throw new Error('The server returned an invalid transcription model list.');
        transcriptionChoices = choices.filter(choice => choice.task === 'transcription');
        transcriptionPicker.replaceChildren(...transcriptionChoices.map(choice => el('option', {
          value: choice.recipe, text: choice.name || fmt.shortID(choice.recipe), disabled: !!choice.refusal, title: choice.refusal || ''
        })));
        if (transcriptionSelected && !transcriptionChoices.some(choice => choice.recipe === transcriptionSelected)) {
          transcriptionPicker.prepend(el('option', { value: transcriptionSelected, text: 'Unavailable model', disabled: true }));
        }
        transcriptionPicker.value = transcriptionSelected || transcriptionChoices.find(choice => !choice.refusal)?.recipe || '';
        transcriptionSelected = transcriptionPicker.value;
        transcriptionNote.textContent = transcriptionRecipe() ? '' : transcriptionChoices.some(choice => !choice.refusal) ? 'The selected model is unavailable. Choose another transcription model.' : 'No transcription model is available. Check models, then retry.';
        transcriptionRetry.hidden = !!transcriptionRecipe();
      }).catch(err => {
        if (disposed || transcriptionLoading !== controller) return;
        transcriptionChoices = []; transcriptionPicker.replaceChildren();
        transcriptionNote.textContent = overgo.friendlyError(err); transcriptionRetry.hidden = false;
      }).finally(() => {
        if (disposed || transcriptionLoading !== controller) return;
        transcriptionLoading = null; transcriptionLoaded = true; renderAttachments();
      });
    }
    transcriptionPicker.addEventListener('change', () => {
      transcriptionSelected = transcriptionPicker.value;
      transcriptionNote.textContent = transcriptionRecipe() ? '' : 'Choose an available transcription model.';
      transcriptionRetry.hidden = !!transcriptionRecipe();
      for (const item of attachments) if (item.kind === 'audio') {
        clearTranscript(item, 'Transcription model changed. Transcribe when ready.');
      }
      renderAttachments();
    });
    function clearTranscript(item, note = '') {
      if (item.transcription) item.transcription.abort();
      item.transcription = null; item.transcribing = false; item.transcriptOffer = null;
      item.transcriptNote = note; item.transcriptAlert = false;
      if (item.offer) { item.offer.replaceChildren(); item.offer.hidden = true; }
    }
    function stopTranscription(item, note) { clearTranscript(item, note); renderAttachments(); }
    function pauseFiles(note) {
      for (const item of attachments) {
        const hadTranscript = item.transcribing || item.transcriptOffer;
        if (item.pending || item.storing) {
          invalidate(item); stage(item, 'cancelled', 'File preparation stopped. Retry or remove the file.');
        }
        if (hadTranscript) clearTranscript(item, note);
      }
      renderAttachments();
    }
    function pagehide() {
      paused = true;
      for (const item of attachments) pausePreview(item);
      if (transcriptionLoading) transcriptionLoading.abort();
      transcriptionLoading = null; transcriptionLoaded = false;
      pauseFiles('Transcription stopped when leaving this page. Transcribe again when ready.');
    }
    function pageshow() { paused = false; renderAttachments(); }
    window.addEventListener('pagehide', pagehide);
    window.addEventListener('pageshow', pageshow);
    function transcribeFile(item) {
      if (disposed || readOnly || paused || !attachments.includes(item) || !item.file || item.transcribing || !transcriptionRecipe()) return;
      clearTranscript(item);
      const controller = new AbortController();
      item.transcription = controller; item.transcribing = true;
      renderAttachments();
      const current = () => !disposed && attachments.includes(item) && item.transcription === controller;
      const form = new FormData();
      form.append('file', item.file, item.name);
      form.append('model', transcriptionRecipe());
      overgo.api.form('/v1/audio/transcriptions', form, { signal: controller.signal }).then((result) => {
        if (!current()) return;
        item.transcription = null; item.transcribing = false;
        offerTranscript(item, String(result.text || ''));
      }, (err) => {
        if (!current()) return;
        item.transcription = null; item.transcribing = false;
        item.transcriptNote = err.name === 'AbortError' ? 'Transcription stopped. Transcribe again or send the recording.' : overgo.friendlyError(err) + ' Transcribe again or send the recording.';
        item.transcriptAlert = err.name !== 'AbortError';
        renderAttachments();
      });
    }
    function offerTranscript(item, text) {
      const offer = {}; item.transcriptOffer = offer;
      const current = () => !disposed && !readOnly && !paused && attachments.includes(item) && item.transcriptOffer === offer;
      const apply = (replace) => {
        if (!current()) return;
        input.value = replace || !input.value ? text : input.value + '\n' + text;
        clearTranscript(item, replace ? 'Draft replaced by the transcript.' : 'Transcript inserted into the draft.');
        input.dispatchEvent(new Event('input')); renderAttachments(); input.focus();
      };
      item.transcriptNote = 'Transcript ready. Insert it after the draft, replace the draft, or discard it.';
      item.offer.replaceChildren(el('span', { class: 'note transcript-text', text: text }),
        el('button', { class: 'link-button', text: 'Insert', 'aria-label': 'Insert the transcript after the draft', onclick: () => apply(false) }),
        el('button', { class: 'link-button', text: 'Replace', 'aria-label': 'Replace the draft with the transcript', onclick: () => apply(true) }),
        el('button', { class: 'link-button', text: 'Discard', 'aria-label': 'Discard the transcript', onclick: () => { if (current()) { clearTranscript(item); renderAttachments(); } } }));
      item.offer.hidden = false;
      renderAttachments();
      item.offer.scrollIntoView({ block: 'nearest' });
    }
    function renderAttachments() {
      if (disposed) return;
      transcriptionHost.hidden = !transcribable() || !attachments.some(item => item.kind === 'audio' && item.file);
      if (!paused && !transcriptionHost.hidden) loadTranscriptionChoices();
      transcriptionPicker.disabled = readOnly || paused || !!transcriptionLoading || !transcriptionChoices.some(choice => !choice.refusal);
      transcriptionRetry.disabled = readOnly || paused;
      transcriptionPicker.parentElement.hidden = !transcriptionChoices.length;
      for (const item of attachments) {
        if (!item.row) {
          item.preview = el('span', { class: 'attachment-preview' }); item.status = el('span', { class: 'note', role: 'status' });
          item.remove = el('button', { class: 'link-button', text: 'Remove', 'aria-label': 'Remove ' + item.name, onclick: () => removeFile(item) });
          item.retry = el('button', { class: 'link-button', text: 'Retry', 'aria-label': 'Retry ' + item.name, onclick: () => { if (!item.file && item.sourceArtifact) reloadStored(item); else prepareFile(item); } });
          item.stop = el('button', { class: 'link-button', text: 'Cancel', 'aria-label': 'Cancel ' + item.name, onclick: () => stopFile(item) });
          item.transcribe = el('button', { class: 'link-button', text: 'Transcribe', 'aria-label': 'Transcribe ' + item.name, onclick: () => transcribeFile(item) });
          item.offer = el('span', { class: 'transcript-offer', hidden: true });
          item.sizeInfo = el('span', { class: 'note', text: overgo.fmt.bytes(item.size) });
          item.metadata = el('details', { class: 'attachment-details' }, el('summary', { text: 'Details' }));
          item.row = el('span', { class: 'attachment-row' }, item.preview, el('span', { class: 'attachment-info' },
            el('span', { class: 'attachment-name', text: item.name }), item.sizeInfo, item.status, item.offer),
            el('span', { class: 'attachment-actions' }, item.transcribe, item.retry, item.stop, item.remove));
        }
        if (item.row.parentNode !== attachmentHost) attachmentHost.appendChild(item.row);
        item.row.dataset.state = item.phase || 'reattach';
        item.row.dataset.size = item.size;
        let status = 'Ready';
        if (item.refusal) status = item.refusal;
        else if (item.needsReattach) status = 'Reattach this file, or remove it to continue.';
        else if (item.storing) status = 'Uploading…';
        else if (item.phase === 'fetching') status = 'Loading stored file…';
        else if (item.phase === 'validating') status = 'Checking image…';
        else if (item.pending) { status = 'Reading…'; if (item.loaded != null) status += ' ' + overgo.fmt.bytes(item.loaded) + ' of ' + overgo.fmt.bytes(item.size); }
        else if (item.transcribing) status = 'Transcribing…';
        else if (item.transcriptNote) status = item.transcriptNote;
        else if (item.artifact) status = 'Stored';
        if (item.status.textContent !== status) item.status.textContent = status;
        item.status.setAttribute('role', item.refusal || item.transcriptAlert ? 'alert' : 'status');
        const focused = document.activeElement;
        item.retry.hidden = (!item.file || !['error', 'cancelled'].includes(item.phase)) && !(item.sourceArtifact && item.needsReattach);
        item.retry.textContent = item.file ? 'Retry' : 'Reload stored file';
        item.stop.hidden = !item.pending && !item.storing && !item.transcribing;
        item.transcribe.hidden = item.kind !== 'audio' || !item.file || item.phase !== 'ready' || item.transcribing || !transcribable();
        item.remove.disabled = item.retry.disabled = item.stop.disabled = readOnly || paused;
        item.transcribe.disabled = readOnly || paused || !!transcriptionLoading || !transcriptionRecipe();
        if (focused === item.retry && item.retry.hidden) (item.stop.hidden ? item.remove : item.stop).focus();
        else if (focused === item.stop && item.stop.hidden) (item.retry.hidden ? item.remove : item.retry).focus();
        const previewKey = !item.pending && !item.refusal && attachmentURL(item);
        if (item.previewKey !== previewKey) { item.previewKey = previewKey; const preview = attachmentPreview(item); item.preview.replaceChildren(...(preview ? [preview] : [])); }
        if (item.artifact && !item.link) { item.link = overgo.artifactLink(item.artifact, 'Stored file'); item.infoLink = el('span', {}, item.link); item.row.querySelector('.attachment-info').appendChild(item.infoLink); }
        if (item.sourceLink && item.sourceArtifact === item.artifact) { item.sourceLink.remove(); item.sourceLink = null; }
        if (item.sourceArtifact && item.sourceArtifact !== item.artifact && !item.sourceLink) { item.sourceLink = overgo.artifactLink(item.sourceArtifact, 'Source file'); item.row.querySelector('.attachment-info').appendChild(item.sourceLink); }
        const standalone = selectedMode() === 'transcription';
        const info = item.row.querySelector('.attachment-info');
        const facts = standalone ? item.metadata : info;
        for (const node of [item.sizeInfo, item.infoLink, item.sourceLink].filter(Boolean)) if (node.parentNode !== facts) facts.appendChild(node);
        if (standalone && item.metadata.parentNode !== info) info.appendChild(item.metadata);
        else if (!standalone) item.metadata.remove();
        item.status.hidden = standalone && ['Ready', 'Stored'].includes(status);
      }
      guidance.hidden = !attachments.length;
      guidanceText.textContent = options.takesAny && options.takesAny() ? 'Files fill the selected model input. The server checks supported types and sizes.' :
        'Supported types: ' + accept.join(', ') + '. Image limit: ' + overgo.fmt.bytes(media.max_image_bytes || 0) + '; other files: ' + overgo.fmt.bytes(media.max_media_bytes || 0) + '.';
      setBusy(busy);
      if (options.onChange) options.onChange();
    }
    function refusal(file, kind) {
      if (file.size === 0) return 'File is empty.';
      const refusals = media.refusals || {};
      if (!accept.includes(file.type)) return refusals[file.type] || refusals[kind] || "the served model does not accept " + (file.type || "this file");
      const limit = kind === "image" ? media.max_image_bytes : media.max_media_bytes;
      return limit && file.size > limit ? "exceeds the " + overgo.fmt.bytes(limit) + " limit" : "";
    }
    function addFile(file, source = {}) {
      if (disposed || readOnly || paused) return;
      const kind = mediaKind(file.type);
      if (selectedMode() === 'transcription' && kind === 'audio') {
        for (const previous of [...attachments]) if (previous.kind === 'audio') removeFile(previous);
      }
      const item = { file, kind, name: file.name, mime: file.type, size: file.size, sourceArtifact: source.artifact, sourceRun: source.run };
      const missing = attachments.findIndex(saved => saved.needsReattach && saved.name === item.name && saved.size === item.size && saved.mime === item.mime);
      if (missing < 0) attachments.push(item); else { removeFile(attachments[missing]); attachments.splice(missing, 0, item); }
      prepareFile(item);
    }
    function reloadStored(item) {
      if (disposed || readOnly || paused || !attachments.includes(item)) return;
      invalidate(item);
      const attempt = Symbol(), controller = new AbortController();
      item.attempt = attempt; item.cancel = () => controller.abort(); item.needsReattach = false;
      const current = () => !disposed && attachments.includes(item) && item.attempt === attempt;
      stage(item, 'fetching'); renderAttachments();
      overgo.api.blob('/artifacts/content?id=' + encodeURIComponent(item.sourceArtifact), { signal: controller.signal }).then(blob => {
        if (!current()) return;
        item.file = new File([blob], item.name, { type: blob.type || item.mime });
        item.mime = item.file.type; item.kind = mediaKind(item.mime); item.size = item.file.size;
        prepareFile(item);
      }, err => { if (current()) { item.needsReattach = true; stage(item, 'error', overgo.friendlyError(err)); renderAttachments(); } });
    }
    function prepareFile(item) {
      if (disposed || readOnly || paused || !attachments.includes(item)) return;
      invalidate(item);
      releaseFile(item);
      const attempt = Symbol(), controller = new AbortController();
      item.attempt = attempt; item.loaded = null;
      const current = () => !disposed && attachments.includes(item) && item.attempt === attempt;
      let reader, probe;
      item.cancel = () => { controller.abort(); if (reader && reader.readyState === FileReader.LOADING) reader.abort(); if (probe) { probe.onload = probe.onerror = null; probe.src = ''; } };
      const fail = err => { if (current()) { stage(item, [400, 413, 415, 422].includes(err.status) ? 'refused' : 'error', overgo.friendlyError(err)); renderAttachments(); } };
      let stored;
      try { stored = options.intake ? options.intake(item.file, { signal: controller.signal }) : null; }
      catch (err) { fail(err); return; }
      if (stored) {
        stage(item, 'uploading'); renderAttachments();
        Promise.resolve(stored).then(result => {
          if (!current()) return;
          item.artifact = result.id; item.release = result.remove; item.target = result.target;
          if (!item.sourceArtifact) item.sourceArtifact = result.id;
          stage(item, 'ready'); renderAttachments();
        }, fail);
        return;
      }
      const reason = refusal(item.file, item.kind);
      if (reason) { stage(item, 'refused', reason); renderAttachments(); return; }
      stage(item, 'reading'); renderAttachments();
      reader = new FileReader();
      reader.onprogress = event => { if (current() && event.lengthComputable) { item.loaded = event.loaded; renderAttachments(); } };
      reader.onerror = () => fail(new Error('Could not read ' + item.name + '. Retry or remove it.'));
      reader.onabort = () => fail(new Error('Reading stopped. Retry or remove the file.'));
      reader.onload = () => {
        if (!current()) return;
        item.dataURL = reader.result;
        if (item.kind === 'image') {
          stage(item, 'validating'); renderAttachments(); probe = new Image();
          probe.onload = () => {
            if (!current()) return;
            let reason = '';
            if (probe.width > media.max_image_dimension || probe.height > media.max_image_dimension) reason = 'Exceeds ' + media.max_image_dimension + ' pixels on a side.';
            else if (probe.width * probe.height > media.max_image_pixels) reason = 'Exceeds ' + media.max_image_pixels + ' pixels.';
            stage(item, reason ? 'refused' : 'ready', reason); renderAttachments();
          };
          probe.onerror = () => fail(new Error('Could not decode ' + item.name + '. Retry or choose another file.'));
          probe.src = item.dataURL;
        } else { stage(item, 'ready'); renderAttachments(); }
      };
      try { reader.readAsDataURL(item.file); } catch (err) { fail(err); }
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
    function setBusy(value, stopping = stop.disabled) {
      const focused = document.activeElement;
      busy = value; stop.disabled = !!(busy && stopping); stop.textContent = stop.disabled ? "Stopping…" : "Stop";
      input.readOnly = readOnly || paused;
      if (attach) attach.disabled = readOnly || paused;
      if (modeSelect) modeSelect.disabled = readOnly || paused;
      send.disabled = readOnly || paused || sendBlocked || busy || attachments.some((item) => item.needsReattach || item.pending || item.refusal || item.storing);
      send.hidden = busy; stop.hidden = !busy;
      if (focused === send && busy) stop.focus();
      else if (focused === stop && !busy) send.focus();
    }
    async function submit() {
      const text = input.value.trim();
      if (!text && !attachments.length) return;
      if (send.disabled) return;
      if (options.onSubmit) await options.onSubmit(text, attachments.slice(), selectedMode());
    }
    send.addEventListener("click", submit);
    stop.addEventListener("click", () => { if (options.onStop) options.onStop(); });
    if (modeSelect && options.onMode) modeSelect.addEventListener("change", () => { capture.close(); options.onMode(modeSelect.value); });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); submit(); }
    });
    input.addEventListener("input", () => { if (options.onChange) options.onChange(); });
    return {
      element, input, attachments, attachmentParts, setBusy, addFile, modeHost, extras,
      setLabels(label, placeholder) { send.textContent = label; input.placeholder = placeholder; },
      setSendBlocked(value) { sendBlocked = value; setBusy(busy); },
      setReadOnly(value) { readOnly = value; if (value) { capture.close(); pauseFiles('Transcription stopped: this conversation became read-only.'); } else renderAttachments(); },
      clearAttachments() { for (const item of attachments) invalidate(item); attachments.length = 0; attachmentHost.replaceChildren(); renderAttachments(); },
      restoreAttachments(items) { attachments.push(...items); renderAttachments(); },
      dispose() { disposed = true; window.removeEventListener('pagehide', pagehide); window.removeEventListener('pageshow', pageshow); if (transcriptionLoading) transcriptionLoading.abort(); capture.dispose(); for (const item of attachments) invalidate(item); },
      closeCapture: capture.close,
      invalidateIntake() {
        capture.close();
        for (const item of attachments) {
          clearTranscript(item, item.transcribing || item.transcriptOffer ? 'Model input changed. Transcribe again when ready.' : '');
          if (item.storing || item.artifact) { invalidate(item); stage(item, 'error', 'Model input changed. Retry to use this file here, or remove it.'); }
        }
        renderAttachments();
      },
      openPicker,
      clearInput() { input.value = ""; },
      mode: selectedMode,
      setMode(id) {
        if (!options.modes?.some(mode => mode.id === id)) return null;
        capture.close();
        if (modeSelect) modeSelect.value = id;
        return options.onMode ? options.onMode(id) : null;
      },
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
