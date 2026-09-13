/* Chat: the conversation the server owns. Every turn is a stored response
   chained through previous_response_id; the rail's selection resumes a
   chain from its record, a turn cut off mid-stream reattaches through
   /interactions/follow, and defaults come from the capability document. */
(function () {
  "use strict";
  const INFLIGHT_STORAGE = "overgo.inflight"; // the response id of a turn this page was streaming
  const DRAFT_STORAGE = "overgo.draft:";
  const drafts = new Map();
  let disposeChat = () => {};
  let saveChat = () => {};
  window.addEventListener("pagehide", () => saveChat());

  // inspectTurn: the side panel over one assistant turn: its run record, then any inspector embedded over it.
  document.addEventListener("keydown", (event) => { // Escape closes the inspector and stops a running turn from anywhere on the page
    const aside = document.getElementById("inspector");
    if (event.key !== "Escape") return;
    if (event.defaultPrevented || document.querySelector("dialog[open]")) return;
    if (aside && !aside.hidden) { aside.hidden = true; aside.replaceChildren(); return; }
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
    if (record.sampling) cards.push(overgo.stat("Sampling", "T " + record.sampling.temperature, record.sampling.samplers.join(" › ") || "no stage"));
    const links = ["model", "recipe", "trace", "receipt", "operation", "run"].filter((name) => record[name]).map((name) => el("span", {}, name + " ", overgo.artifactLink(record[name])));
    const host = el("div");
    const seed = { prompt: (record.prompt || "") + (record.completion ? "\n" + record.completion : "") };
    const inspectors = [["lens", "Logits", seed], ["states", "States", seed], ["attention", "Attention", seed], ["model", "Model"], ["vocab", "Vocabulary"], ["tensors", "Tensors"]];
    body.append(el("div", { class: "statgrid" }, ...cards), el("div", { class: "row artifact-links" }, ...links),
      el("div", { class: "row" }, ...inspectors.map(([id, label, given]) => el("button", { class: "btn alt", text: label, onclick: () => overgo.embed(id, host, given) }))), host);
  };

  window.overgo.registerTab({
    id: "chat",
    onDeactivate() { saveChat(); },
    async mount(panel, overgo) {
      disposeChat();
      let disposed = false;
      let saveDraft = () => {};
      let updateSwitch = () => {};
      let disposeComposer = () => {};
      let controller = null;
      let activeTurn = null;
      let activeOperation = null;
      let recoveryRow = null;
      let actionPending = false;
      const selected = overgo.conversation();
      let conversationRoot = selected ? selected.root : "";
      disposeChat = () => {
        saveDraft();
        disposed = true;
        disposeComposer();
        document.removeEventListener("overgo-model-switch", updateSwitch);
        // A queued explicit Stop still needs the incoming ID. Keep only that
        // acknowledgement alive across navigation, then cancel its execution.
        if (controller && !(activeTurn && activeTurn.stopRequested && !activeTurn.response) && !(activeOperation && activeOperation.stopRequested && !activeOperation.id)) controller.abort();
      };
      const { el, clear, fmt } = overgo;
      clear(panel);
      const capabilities = overgo.capabilities();
      if (!capabilities) { panel.appendChild(overgo.errorBanner("the served model declares no capabilities yet")); return; }
      const modelID = capabilities.id;
      const served = overgo.servedModel();
      const servedModel = capabilities.model || (served && served.model);
      const mismatchedModel = item => item && ((item.model && servedModel && item.model !== servedModel) || (item.recipe && capabilities.recipe && item.recipe !== capabilities.recipe));
      let otherModel = mismatchedModel(selected);
      const params = capabilities.generation;
      const contextLength = capabilities.context_length;
      const tokenLimit = capabilities.max_output_tokens;
      // Without a declared recipe, never reuse a draft under a display name.
      const draftScope = capabilities.recipe || crypto.randomUUID();
      const draftKey = () => DRAFT_STORAGE + JSON.stringify([draftScope, conversationRoot]);
      let storageUnavailable = !capabilities.recipe;
      let draft = drafts.get(draftKey());
      if (!draft) {
        try { if (capabilities.recipe) draft = JSON.parse(sessionStorage.getItem(draftKey()) || "null"); }
        catch (_) { storageUnavailable = true; }
      }
      if (!draft || typeof draft !== "object") draft = {};
      let uncertainPrevious = typeof draft.uncertainPrevious === 'string' ? draft.uncertainPrevious : null;
      let uncertainText = typeof draft.uncertainText === 'string' ? draft.uncertainText : '';
      let uncertainKind = draft.uncertainKind === 'operation' ? 'operation' : 'response';
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
      const temperature = el("input", { class: "keyfield w-80", type: "number", min: 0, step: "any", value: params.temperature, "aria-label": "temperature", "data-draft-setting": "temperature" });
      const maxTokens = el("input", { class: "keyfield w-90", type: "number", min: 1, step: 1, value: Math.min(params.max_tokens, tokenLimit), max: tokenLimit, "aria-label": "max tokens", "data-draft-setting": "tokens" });
      system.dataset.draftSetting = "system";
      system.value = typeof draft.system === "string" ? draft.system : "";
      if (draft.temperature !== "" && Number.isFinite(Math.fround(Number(draft.temperature))) && Number(draft.temperature) >= 0) temperature.value = draft.temperature;
      if (draft.tokens !== "" && Number.isFinite(Number(draft.tokens)) && Number(draft.tokens) >= 1) maxTokens.value = Math.min(Math.floor(Number(draft.tokens)), tokenLimit);

      const settings = document.getElementById("conversation-settings");
      const focusedSetting = settings.contains(document.activeElement) && document.activeElement.dataset.draftSetting;
      settings.replaceChildren(el("section", { class: "settings-section" }, el("h3", { text: "Conversation" }),
        el("label", { class: "setting-field" }, "System prompt", system),
        el("div", { class: "settings-fields" }, el("label", { class: "setting-field" }, "Temperature", temperature), el("label", { class: "setting-field" }, "Maximum output tokens", maxTokens)),
        el("details", {}, el("summary", { text: "Last response usage" }), facts)));
      const exportStatus = el('div', { class: 'note', role: 'status', hidden: true });
      settings.firstChild.append(el('details', {}, el('summary', { text: 'Copy or export conversation' }),
        el('div', { class: 'row' }, el('button', { class: 'link-button conversation-read-action', text: 'Copy conversation', onclick: () => exportConversation(true) }),
          el('button', { class: 'link-button conversation-read-action', text: 'Export Markdown', onclick: () => exportConversation(false) })), exportStatus));
      async function exportConversation(copy) {
        if (disposed || controller || activeTurn || actionPending) return;
        actionPending = true; updateBusy(); exportStatus.hidden = false; exportStatus.textContent = 'Preparing conversation…'; exportStatus.setAttribute('role', 'status');
        try {
          const selected = overgo.conversation();
          let messages = thread.messages;
          if (selected) messages = (await overgo.api.get('/interactions/messages?response=' + encodeURIComponent(selected.latest))).messages;
          if (disposed) return;
          const visible = messages.filter(message => ['user', 'assistant'].includes(message.role));
          if (!visible.length) throw new Error('There are no conversation messages to export yet.');
          const title = selected && selected.title || 'Conversation';
          const markdown = '# ' + title.replace(/[\r\n]/g, ' ') + '\n\n' + visible.map(message => {
            let source = '';
            if (message.response) source = new URL('/interactions/inspect?response=' + encodeURIComponent(message.response), location.origin).href;
            const media = (message.media || []).map(item => '\n\nAttachment: ' + item.type + ' — [stored turn](' + source + ')').join('');
            return '## ' + (message.role === 'user' ? 'User' : 'Assistant') + '\n\n' + message.content + media;
          }).join('\n\n') + '\n';
          if (copy) await navigator.clipboard.writeText(markdown);
          else {
            overgo.downloadBlob(exportStatus, new Blob([markdown], { type: 'text/markdown;charset=utf-8' }), 'conversation-' + (selected && selected.latest || 'draft') + '.md');
          }
          exportStatus.textContent = copy ? 'Conversation copied.' : 'Markdown download started.';
        } catch (err) { if (!disposed) { exportStatus.setAttribute('role', 'alert'); exportStatus.textContent = overgo.friendlyError(err) + ' Try again.'; } }
        finally { actionPending = false; if (!disposed) updateBusy(); }
      }
      // artifactField: the mode's artifact-typed control, if any. A media card re-enters the composer as the next
      // turn's attachment (refused or accepted by the served capability) or, in such a mode, as the control's stored id;
      // a fresh attachment in such a mode stores through the intake route and fills the control.
      // A file goes to the slot of its kind (the slot's declared media), else to the first slot.
      const generation = { capabilities: null, capability: null, fields: new Map() };
      const artifactField = (file) => { const slots = [...generation.fields.values()].filter((field) => field.control.type === "artifact");
        return slots.find((field) => file && field.control.media && file.type.startsWith(field.control.media + "/")) || slots[0]; };
      const fieldUploads = new WeakMap();
      const thread = overgo.thread(panel, { reuse: (file, artifact, source) => { const field = artifactField(file); if (field && artifact) field.input.value = artifact; else composer.addFile(file, source); },
        replay: replayRecord, lineage: showLineage, actions: messageActions, marker: capabilities.remote ? "remote" : "" });
      const branchNotice = el('div', { class: 'note', role: 'status', hidden: true });
      function actionsBlocked() { return disposed || !!controller || !!activeTurn || actionPending || uncertainPrevious !== null || otherModel || overgo.modelSwitching(); }
      function messageActions(message) {
        if (!['user', 'assistant'].includes(message.role)) return [];
        return [el('button', { class: 'link-button turn-action', title: 'Create a new branch using current settings', text: message.role === 'user' ? 'Edit and resend' : 'Regenerate',
          disabled: actionsBlocked(), onclick: () => prepareBranch(message) })];
      }
      function renderConversation(messages) {
        thread.reset();
        for (const message of messages) if (['user', 'assistant'].includes(message.role)) {
          const shown = thread.add(message.role, message.content);
          Object.assign(shown, { response: message.response, media: message.media, tool_calls: message.tool_calls });
          thread.renderMessage(shown, false);
        }
      }
      function restoreBranch(branch, failure) {
        conversationRoot = branch.root; lastResponseID = branch.previousSelection;
        overgo.rememberConversation(branch.source); renderConversation(branch.original);
        branchNotice.hidden = true; saveDraft();
        thread.errorRow(failure + ' Original conversation restored.');
        thread.node.appendChild(el('button', { class: 'link-button turn-action', text: 'Retry branch', disabled: actionsBlocked(), onclick: () => submit(branch.text, [], 'chat', branch) }));
      }
      // Preserve the stored media positions when regenerating. An edited prompt
      // keeps its attachments after the new text, as the inline editor states.
      function branchInput(message, edited) {
        const content = [], bytes = new TextEncoder().encode(message.content), decoder = new TextDecoder();
        let offset = 0;
        if (edited !== undefined) content.push({ type: 'input_text', text: edited });
        for (const media of message.media || []) {
          if (edited === undefined) {
            content.push({ type: 'input_text', text: decoder.decode(bytes.slice(offset, media.text_offset)) });
            offset = media.text_offset;
          }
          if (media.type === 'image') content.push({ type: 'input_image', image_url: media.data });
          else if (media.type === 'audio') content.push({ type: 'input_audio', input_audio: media.format ? { data: media.data, format: media.format } : { url: media.data } });
          else if (media.type === 'video') content.push({ type: 'input_video', input_video: { data: media.data, fps: media.fps || 0 } });
          else throw new Error('This attachment cannot be resent. Start a new conversation and reattach it.');
        }
        if (edited === undefined) content.push({ type: 'input_text', text: decoder.decode(bytes.slice(offset)) });
        return { role: 'user', content };
      }
      async function prepareBranch(message) {
        if (actionsBlocked()) return;
        const editor = el('form', { class: 'turn-editor' }, el('div', { class: 'note', role: 'status', text: 'Loading the stored turn…' }));
        panel.querySelectorAll('.turn-editor').forEach(node => node.remove());
        message.node.appendChild(editor); actionPending = true; updateBusy();
        try {
          const chain = await overgo.api.get('/interactions/messages?response=' + encodeURIComponent(message.response));
          if (disposed) return;
          if (mismatchedModel(chain)) throw new Error('Choose this conversation’s original model to resend it.');
          if (chain.status === 'in_progress') throw new Error('This response is still running. Open it to resume or stop it.');
          const turn = chain.messages.filter(item => item.response === message.response);
          const users = turn.filter(item => item.role === 'user');
          if (!users.length || turn.some(item => item.role === 'tool' || (item.tool_calls || []).length)) throw new Error('This turn includes tool execution. Start a new message to continue it.');
          const editIndex = message.role === 'user' ? thread.messages.filter(item => item.role === 'user' && item.response === message.response).indexOf(message) : users.length - 1;
          if (editIndex < 0 || !users[editIndex]) throw new Error('The stored turn changed. Reopen the conversation and try again.');
          const branch = { prefix: chain.messages.filter(item => item.response !== message.response), previous: chain.previous || '', text: users[editIndex].content,
            users: users.map(item => ({ ...item })), input: users.map(item => branchInput(item)) };
          const run = () => { if (!validateSettings()) return; editor.remove(); return submit(branch.text, [], 'chat', branch); };
          if (message.role === 'assistant') { actionPending = false; updateBusy(); await run(); return; }
          const field = el('textarea', { class: 'text', 'aria-label': 'Edit message', value: branch.text, required: true });
          const cancel = () => { editor.remove(); message.node.querySelector('.turn-action')?.focus(); };
          editor.replaceChildren(field, el('div', { class: 'note', text: 'Creates a new branch using current settings. Existing attachments follow the edited text.' }),
            el('div', { class: 'row' }, el('button', { class: 'btn', type: 'submit', text: 'Send edited message' }), el('button', { class: 'link-button', type: 'button', text: 'Cancel edit', onclick: cancel })));
          editor.addEventListener('submit', event => { event.preventDefault(); if (!field.value.trim() || actionsBlocked()) return;
            branch.text = field.value.trim(); branch.users[editIndex].content = branch.text; branch.input[editIndex] = branchInput(users[editIndex], branch.text); run(); });
          editor.addEventListener('keydown', event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); cancel(); } });
          field.focus();
        } catch (err) { if (!disposed) editor.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
        finally { actionPending = false; if (!disposed) updateBusy(); }
      }
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
      async function requestOf(artifact, run, options) {
        const lineage = await overgo.api.get("/artifacts/lineage?id=" + encodeURIComponent(artifact), options);
        const made = lineage.producers.find((item) => item.run === run) || lineage.producers[0];
        const input = made && made.inputs.find((item) => (item.schema || "").endsWith("-input.v1"));
        return input && overgo.api.get("/artifacts/content?id=" + encodeURIComponent(input.id), options);
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
        if (actionsBlocked()) return;
        controller = new AbortController();
        updateBusy();
        try {
          const options = { signal: controller.signal };
          if (!generation.capabilities) generation.capabilities = await overgo.api.get("/generation/capabilities", options);
          const run = await overgo.api.get("/runs?id=" + encodeURIComponent(event.run), options);
          const capability = generation.capabilities.find((item) => item.recipe === run.recipe && !item.refusal);
          if (!capability) throw new Error("the capability that made it is no longer active");
          const request = await requestOf(event.artifact, run.id, options);
          if (!request) throw new Error("the run records no request");
          controller.signal.throwIfAborted();
          if (label === "vary") {
            if (!capability.controls.some((control) => control.name === "seed")) throw new Error("the request declares no seed to vary");
            request.seed = crypto.getRandomValues(new Uint32Array(1))[0];
          }
          thread.add("user", label + " " + fmt.shortID(event.artifact));
          const work = trackOperation();
          await thread.consume(overgo.streams.replay(capability, request, controller.signal, event.artifact, label, work.lifecycle));
        } catch (err) {
          if (disposed) return;
          thread.errorRow(err.name === "AbortError" ? "Request disconnected; check Activity for its result." : overgo.friendlyError(err));
          if (err.operationUnconfirmed) { uncertainKind = 'operation'; uncertainPrevious = lastResponseID; uncertainText = label + ' ' + event.artifact; saveDraft(); showUncertainty(); }
        }
        finally { if (activeOperation && !activeOperation.id) activeOperation.row.remove(); activeOperation = null; controller = null; if (!disposed) updateBusy(); }
      }
      function updateBusy() {
        composer.setBusy(!!controller || !!activeTurn, (activeTurn && activeTurn.stopRequested) || (activeOperation && activeOperation.stopRequested));
        composer.setSendBlocked(actionPending || uncertainPrevious !== null || overgo.modelSwitching());
        panel.querySelectorAll('.turn-action, .media-replay, .turn-editor button[type=submit]').forEach(button => { button.disabled = actionsBlocked(); });
        settings.querySelectorAll('.conversation-read-action').forEach(button => { button.disabled = disposed || !!controller || !!activeTurn || actionPending; });
      }
      function forgetTurn(turn) {
        try {
          const saved = JSON.parse(sessionStorage.getItem(INFLIGHT_STORAGE) || "null");
          if (saved && saved.response === turn.response && (!saved.model || saved.model === turn.model)) sessionStorage.removeItem(INFLIGHT_STORAGE);
        } catch (_) { /* storage unavailable */ }
      }
      function restorePrompt(prompt) {
        if (!prompt || disposed || composer.input.value || composer.attachments.length) return;
        composer.input.value = prompt.text;
        composer.restoreAttachments(prompt.attachments || []);
        saveDraft();
      }
      function showRecovery(err, turn) {
        if (recoveryRow) recoveryRow.remove();
        recoveryRow = el("div", { class: "note", role: "status" },
          overgo.friendlyError(err) + " ",
          el("button", { class: "btn alt", text: "Resume response", onclick: () => resumeTurn(turn) }));
        thread.node.appendChild(recoveryRow);
      }
      async function cancelTurn(turn) {
        if (!turn.response || turn.cancelSent) return;
        turn.cancelSent = true;
        try {
          await overgo.api.post("/interactions/cancel", { response: turn.response, model: turn.model });
          if (!disposed && activeTurn === turn && !controller && !turn.stopFollowed) {
            turn.stopFollowed = true;
            await resumeTurn(turn);
          }
        } catch (err) {
          turn.cancelSent = turn.stopRequested = false;
          if (!disposed && activeTurn === turn) {
            thread.errorRow("Could not stop the response: " + overgo.friendlyError(err) + ". Try Stop again.");
            updateBusy();
          }
        }
      }
      overgo.stopTurn = () => {
        if (disposed) return;
        if (activeTurn) {
          activeTurn.stopRequested = true;
          updateBusy();
          cancelTurn(activeTurn);
        } else if (activeOperation) { activeOperation.stopRequested = true; updateBusy(); cancelBackgroundOperation(activeOperation); }
        else if (controller) controller.abort();
      };
      async function cancelBackgroundOperation(work) {
        if (!work.id || work.cancelSent) return;
        work.cancelSent = true; work.failure = '';
        try { await overgo.api.post('/operations/cancel', { id: work.id }); }
        catch (err) {
          work.cancelSent = work.stopRequested = false;
          if (!disposed && activeOperation === work) { work.failure = 'Could not stop: ' + overgo.friendlyError(err) + '. Try Stop again.'; work.status.textContent = work.failure; updateBusy(); }
        }
      }
      function trackOperation() {
        const status = el('span', { text: 'Starting operation…' }), row = el('div', { class: 'note', role: 'status' }, status);
        const work = activeOperation = { id: '', status, row };
        thread.node.appendChild(row);
        work.lifecycle = {
          accepted: async id => {
            work.id = id;
            if (!disposed) row.append(' ', el('button', { class: 'link-button', text: 'View activity', onclick: () => overgo.showOperation(id) }));
            if (work.stopRequested) await cancelBackgroundOperation(work);
            if (disposed && controller) controller.abort();
          },
          observe: current => { if (!disposed) status.textContent = work.failure && !['completed', 'failed', 'cancelled'].includes(current.state) ? work.failure : current.state + (current.progress && current.progress.total != null ? ' · ' + current.progress.completed + ' / ' + current.progress.total : ''); },
        };
        return work;
      }
      const composer = overgo.composer(panel, {
        onSubmit: submit,
        onStop: overgo.stopTurn,
        onChange: () => { composer.input.setCustomValidity(''); saveDraft(); },
        takesAny: () => !!artifactField(),
        captureAccept: () => {
          const slots = [...generation.fields.values()].filter(field => field.control.type === "artifact");
          return slots.length ? (capabilities.media.intake_accept || []).filter(mime => slots.some(field => !field.control.media || mime.startsWith(field.control.media + "/"))) : capabilities.media.accept || [];
        },
        intake: (file, { signal }) => {
          const field = artifactField(file);
          if (!field) return null;
          const attempt = {}, previousValue = field.input.value; fieldUploads.set(field, attempt);
          return overgo.api.upload('/artifacts/intake', file, { signal }).then(stored => {
            signal.throwIfAborted();
            if (!stored || typeof stored.id !== 'string' || !stored.id) throw new Error('Upload returned no stored file identity. Retry or remove the file.');
            if (disposed || !field.input.isConnected || ![...generation.fields.values()].includes(field)) throw new Error('Model input changed. Retry to use the file here.');
            if (fieldUploads.get(field) !== attempt) throw new Error('A newer file was selected for this input. Retry or remove this file.');
            if (field.input.value !== previousValue) throw new Error('The input was changed while uploading. Retry or remove this file.');
            field.input.value = stored.id; field.input.dispatchEvent(new Event('intake'));
            return { id: stored.id, target: field.input, remove: () => { if (field.input.value === stored.id) { field.input.value = ''; field.input.dispatchEvent(new Event('input', { bubbles: true })); } } };
          });
        },
        modes: (capabilities.modes || []).filter((mode) => mode.enabled), // the served recipe declares agent mode with the rest
        onMode: (mode) => { agentHost.hidden = mode !== "agent"; saveDraft(); return renderMode(mode); },
        controls: [],
      });
      disposeComposer = () => composer.dispose();
      composer.input.value = typeof draft.text === "string" ? draft.text : "";
      if (Array.isArray(draft.attachments)) composer.restoreAttachments(draft.attachments.filter(item => item && typeof item.name === "string" && typeof item.mime === "string" && Number.isFinite(item.size) && item.size >= 0).map(item => ({ name: item.name, mime: item.mime, size: item.size, kind: overgo.mediaKind(item.mime), needsReattach: true,
        sourceArtifact: typeof item.sourceArtifact === 'string' ? item.sourceArtifact : undefined, sourceRun: typeof item.sourceRun === 'string' ? item.sourceRun : undefined })));
      const draftNotice = el("div", { class: "note", role: "status", hidden: true });
      composer.extras.appendChild(draftNotice);
      const uncertaintyNotice = el('div', { class: 'note', role: 'alert', hidden: true });
      composer.extras.appendChild(uncertaintyNotice);
      function showUncertainty() {
        uncertaintyNotice.hidden = uncertainPrevious === null;
        if (uncertaintyNotice.hidden) { uncertaintyNotice.replaceChildren(); return; }
        uncertaintyNotice.replaceChildren('The request outcome is unknown. Check ' + (uncertainKind === 'operation' ? 'activity' : 'conversation history') + ' before sending again. ',
          el('button', { class: 'link-button', text: uncertainKind === 'operation' ? 'Review activity' : 'Review history', onclick: () => { if (uncertainKind === 'operation') overgo.showOperation(''); else { overgo.refreshConversations(); document.getElementById('navigation-toggle').click(); } } }),
          el('button', { class: 'link-button', text: 'Allow sending again', onclick: () => { uncertainPrevious = null; uncertainText = ''; saveDraft(); showUncertainty(); updateBusy(); composer.input.focus(); } }),
          el('details', {}, el('summary', { text: 'Unconfirmed message' }), el('div', { class: 'message-text', text: uncertainText })));
      }
      showUncertainty();
      const modelNotice = el("div", { class: "note", role: "status", hidden: true }, "Confirming the selected model… Choose a model again if loading fails.");
      composer.extras.appendChild(modelNotice);
      composer.extras.appendChild(branchNotice);
      function persistDraft(key, value) {
        drafts.set(key, value);
        try { if (capabilities.recipe) sessionStorage.setItem(key, JSON.stringify(value)); }
        catch (_) { storageUnavailable = true; }
        draftNotice.hidden = !storageUnavailable || !(value.text || value.system || value.attachments.length || value.temperature !== String(params.temperature) || value.tokens !== String(Math.min(params.max_tokens, tokenLimit)));
        draftNotice.textContent = capabilities.recipe ? "Draft kept for this page only. Browser storage is unavailable; reloading may lose it." : "Draft cannot be restored: the server has not declared this model's recipe.";
      }
      saveDraft = () => {
        if (disposed) return;
        persistDraft(draftKey(), { text: composer.input.value, system: system.value, temperature: temperature.value, tokens: maxTokens.value, mode: composer.mode(),
          uncertainPrevious: uncertainPrevious === null ? undefined : uncertainPrevious, uncertainText: uncertainPrevious === null ? undefined : uncertainText, uncertainKind: uncertainPrevious === null ? undefined : uncertainKind,
          attachments: composer.attachments.map(item => ({ name: item.name, mime: item.mime, size: item.size, sourceArtifact: item.sourceArtifact, sourceRun: item.sourceRun })) });
      };
      saveChat = saveDraft;
      for (const field of [system, temperature, maxTokens]) { field.addEventListener("input", saveDraft); field.addEventListener("change", saveDraft); }
      updateSwitch = () => { modelNotice.hidden = !overgo.modelSwitching(); updateBusy(); };
      document.addEventListener("overgo-model-switch", updateSwitch);
      updateSwitch();
      saveDraft();
      function moveDraft(root) {
        if (conversationRoot === root) return;
        saveDraft();
        persistDraft(draftKey(), { ...drafts.get(draftKey()), text: "", attachments: [] });
        conversationRoot = root;
        saveDraft();
      }
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
      const galleryLimit = 12; // the newest outputs a mode's gallery rail lists
      async function renderMode(mode) {
        composer.invalidateIntake();
        generation.capability = null;
        generation.fields.clear(); generation.picker = null;
        composer.modeHost.replaceChildren();
        if (!mode || mode === "chat" || mode === "agent") return;
        try {
          if (!generation.capabilities) generation.capabilities = await overgo.api.get("/generation/capabilities");
        } catch (err) { composer.modeHost.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); return; }
        if (disposed || composer.mode() !== mode) return;
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
          composer.invalidateIntake();
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
        const content = [];
        for (const part of parts) {
          if (part.type === "image_url") content.push({ type: "input_image", image_url: part.image_url.url });
          else if (part.type === "input_audio") content.push({ type: "input_audio", input_audio: part.input_audio });
          else if (part.type === "input_video") content.push({ type: "input_video", input_video: part.input_video });
          else if (part.type === "input_file") content.push(part);
        }
        content.push({ type: "input_text", text });
        return [{ role: "user", content: parts.length ? content : text }];
      }

      function streamTurn(path, body, method) { return overgo.api.stream(path, body, { signal: controller.signal, method }); }

      // A recovery handle belongs to this mounted conversation and model.
      // Terminal events alone clear it; connection loss keeps Resume available.
      async function consumeTurn(response, assistant, inputTokens) {
        const turn = activeTurn;
        let latest = turn.response || "";
        async function* noting(events) {
          for await (const event of events) {
            if (disposed) {
              if (event.type === "created" && turn.stopRequested) {
                turn.response = event.id;
                await cancelTurn(turn);
                if (controller) controller.abort();
              }
              return;
            }
            if (event.id && (event.type === "created" || event.status)) {
              if (turn.response && turn.response !== event.id) throw new Error("Recovery returned a different response.");
              latest = turn.response = event.id;
            }
            if (event.type === "created") {
              if (turn.branch && !turn.previous) { saveDraft(); conversationRoot = event.id; saveDraft(); }
              else if (!conversationRoot) moveDraft(event.id);
              const root = conversationRoot;
              overgo.rememberConversation({ root, latest: event.id, model: servedModel, recipe: capabilities.recipe });
              for (const user of turn.users || []) { user.response = event.id; thread.renderMessage(user, false); }
              try { sessionStorage.setItem(INFLIGHT_STORAGE, JSON.stringify({ root, response: event.id, model: turn.model })); } catch (_) { /* storage unavailable */ }
              if (turn.stopRequested) cancelTurn(turn);
            }
            yield event;
          }
        }
        const terminal = await thread.consume(noting(overgo.streams.responses(response)), assistant);
        if (disposed) return;
        if (!terminal.status) throw new Error("The response has no confirmed result. Resume to check it.");
        lastResponseID = terminal.status === "failed" ? turn.previous : latest;
        if (terminal.status === "failed" && !turn.previous && !turn.branch) { moveDraft(""); overgo.rememberConversation(null); }
        if (latest && assistant) { assistant.response = latest; thread.renderMessage(assistant, false); }
        forgetTurn(turn);
        activeTurn = null;
        if (recoveryRow) { recoveryRow.remove(); recoveryRow = null; }
        if (terminal.status === "failed") {
          if (turn.branch) restoreBranch(turn.branch, 'The branch failed.'); else restorePrompt(turn.prompt);
        }
        renderFacts(inputTokens, terminal.usage, terminal.timings);
        overgo.refreshConversations();
      }

      function turnFailure(err, turn) {
        if (disposed) return;
        if (turn && turn.response && err.status !== 404) {
          showRecovery(err, turn);
          return;
        }
        const unknown = turn && !turn.response && !err.status;
        if (unknown) { uncertainKind = 'response'; uncertainPrevious = turn.branch ? turn.branch.previousSelection : turn.previous; uncertainText = turn.branch ? turn.branch.text : turn.prompt.text; }
        if (turn) {
          forgetTurn(turn);
          activeTurn = null;
          lastResponseID = turn.previous;
          if (!turn.branch) {
            if (!lastResponseID) { moveDraft(""); overgo.rememberConversation(null); }
            restorePrompt(turn.prompt);
          }
        }
        if (turn && turn.branch) restoreBranch(turn.branch, unknown ? 'The branch outcome is unknown.' : overgo.friendlyError(err));
        else if (!unknown) thread.errorRow(overgo.friendlyError(err));
        if (unknown) { saveDraft(); showUncertainty(); }
        overgo.refreshConversations();
      }

      async function resumeTurn(turn) {
        if (disposed || controller || activeTurn !== turn) return;
        controller = new AbortController();
        updateBusy();
        try {
          const response = await streamTurn("/interactions/follow?response=" + encodeURIComponent(turn.response) + "&model=" + encodeURIComponent(turn.model), null, "GET");
          // Follow replays from the beginning. Replace this assistant's partial
          // text only after connection succeeds, so deltas are never doubled.
          turn.assistant.content = "";
          thread.renderMessage(turn.assistant, true);
          await consumeTurn(response, turn.assistant, null);
        } catch (err) { turnFailure(err, turn); }
        finally { controller = null; if (!disposed) updateBusy(); }
      }

      function validateSettings() {
        const invalid = [temperature, maxTokens].find(field => !field.checkValidity());
        if (!invalid) return true;
        if (!document.getElementById('settings-dialog').open) document.getElementById('settings-toggle').click();
        invalid.focus(); invalid.reportValidity(); return false;
      }
      async function submit(text, attachments, mode, branch) {
        if (actionsBlocked() || !validateSettings()) return;
        if (mode && !['chat', 'agent', 'embeddings', 'rerank'].includes(mode)) {
          if (!generation.capability) { thread.errorRow('Choose a model for this mode before sending.'); return; }
          const invalid = [...generation.fields.values()].find(field => !field.input.checkValidity());
          if (invalid) { invalid.input.scrollIntoView({ block: 'nearest' }); invalid.input.focus(); invalid.input.reportValidity(); return; }
          const body = overgo.bodyControl(generation.capability.controls);
          if (body && body.required && !text.trim()) { composer.input.setCustomValidity('Enter ' + (body.label || body.name) + '.'); composer.input.reportValidity(); return; }
        }
        welcome.remove();
        const parts = branch ? [] : composer.attachmentParts();
        if (branch) {
          // The reader may have sent another message while this editor was
          // open. Capture the return point at submission, not at edit start.
          branch = { ...branch, source: { ...overgo.conversation() }, root: conversationRoot, previousSelection: lastResponseID,
            original: thread.messages.map(item => ({ role: item.role, content: item.content, response: item.response, media: item.media, tool_calls: item.tool_calls })) };
          renderConversation(branch.prefix); lastResponseID = branch.previous;
          branchNotice.replaceChildren('New branch. ', el('button', { class: 'link-button', text: 'Return to original', onclick: () => overgo.openConversation(branch.source) }));
          branchNotice.hidden = false;
        } else { composer.clearInput(); composer.clearAttachments(); }
        const users = (branch ? branch.users : [{ content: overgo.userLine(text, attachments) }]).map(item => {
          const shown = thread.add('user', item.content); shown.media = item.media; return shown;
        });
        controller = new AbortController();
        updateBusy();
        if (branch) composer.element.querySelector('.stop-button').focus();
        let assistant = null;
        try {
          // A mode beyond chat is one generation over the shared dispatch; its
          // media lands in this thread as artifacts with their provenance.
          if (mode === "agent") {
            const history = (histories.get(agentPicker.value) || []).concat([{ role: "user", content: parts.length ? [...parts, { type: "text", text }] : text }]);
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
            const work = generation.capability ? trackOperation() : null;
            const selection = generation.capability ? { capability: generation.capability, fields: generation.fields, sources, lifecycle: work.lifecycle } : null;
            const terminal = await thread.consume(overgo.generate(mode, text, parts, controller.signal, selection));
            if (!disposed && terminal.status !== 'completed') {
              restorePrompt({ text, attachments });
              if (terminal.status === 'unknown') { uncertainKind = 'operation'; uncertainPrevious = lastResponseID; uncertainText = text; saveDraft(); showUncertainty(); }
            }
            if (work && !work.id) work.row.remove();
            return;
          }
          assistant = thread.add("assistant", "");
          const request = { model: modelID, input: branch ? branch.input : responsesInput(text, parts), stream: true, store: true };
          if (lastResponseID) request.previous_response_id = lastResponseID;
          if (system.value.trim()) request.instructions = system.value.trim();
          if (temperature.value !== "") request.temperature = Number(temperature.value);
          if (maxTokens.value !== "") request.max_output_tokens = Number(maxTokens.value);
          const count = await overgo.api.post("/v1/responses/input_tokens", request, { signal: controller.signal }).catch(() => null) /* reviewed: a hosted model's provider tokenizes, the count route refuses, and the meter stays empty by design */;
          controller.signal.throwIfAborted();
          activeTurn = { response: "", model: modelID, previous: lastResponseID, assistant, users, branch, prompt: { text, attachments } };
          renderFacts(count ? count.input_tokens : null, null, null);
          await consumeTurn(await streamTurn("/v1/responses", request, "POST"), assistant, count ? count.input_tokens : null);
        } catch (err) {
          if (disposed) return;
          if (!mode || mode === "chat") {
            if (err.name === "AbortError" && !activeTurn) thread.errorRow("Stopped before sending.");
            else turnFailure(err, activeTurn);
            if (!activeTurn && !branch) restorePrompt({ text, attachments });
            if (branch && !activeTurn && err.name === 'AbortError') restoreBranch(branch, 'Stopped before sending.');
          } else { thread.errorRow(err.name === "AbortError" ? "Request disconnected; check Activity for its result." : overgo.friendlyError(err)); restorePrompt({ text, attachments }); }
        } finally {
          activeOperation = null;
          controller = null;
          if (!disposed) {
            updateBusy();
            // Stop can interrupt the original socket to release a stalled
            // server write. Reattach once to read its terminal state.
            if (activeTurn && activeTurn.response && activeTurn.stopRequested && activeTurn.cancelSent && !activeTurn.stopFollowed) {
              activeTurn.stopFollowed = true;
              await resumeTurn(activeTurn);
            }
          }
        }
      }

      // Server state can recover a running conversation even when another
      // conversation has replaced the tab's optional storage handle.
      if ((capabilities.modes || []).some(mode => mode.enabled && mode.id === draft.mode)) await composer.setMode(draft.mode);
      if (disposed) return;
      let saved = null;
      try { saved = JSON.parse(sessionStorage.getItem(INFLIGHT_STORAGE) || "null"); } catch (_) { /* storage unavailable */ }
      let inflight = selected && saved && selected.root === saved.root && (!saved.model || saved.model === modelID) ? saved.response : "";
      let chain = null;
      if (selected) {
        composer.setBusy(true);
        try {
          chain = await overgo.api.get("/interactions/messages?response=" + encodeURIComponent(selected.latest));
          if (disposed) return;
          otherModel = mismatchedModel({ ...selected, ...chain });
          if (chain.status === "in_progress") inflight = chain.response;
          welcome.remove();
          for (const message of chain.messages || []) {
            if (message.role !== "user" && message.role !== "assistant") continue;
            if (inflight && message.role === "assistant" && message.response === inflight) continue;
            const shown = thread.add(message.role, message.content);
            shown.response = message.response;
            shown.media = message.media; shown.tool_calls = message.tool_calls;
            thread.renderMessage(shown, false);
          }
          lastResponseID = chain.status === "failed" ? (chain.previous || "") : chain.response;
          if (uncertainPrevious !== null && uncertainKind !== 'operation' && chain.response !== uncertainPrevious) { uncertainPrevious = null; uncertainText = ''; saveDraft(); showUncertainty(); }
          if (chain.status === "failed" && !lastResponseID) { moveDraft(""); overgo.rememberConversation(null); }
          if (!inflight && chain.status === 'cancelled') thread.node.appendChild(el('div', { class: 'note', role: 'status', text: 'Stopped' }));
          else if (!inflight && chain.failure) thread.errorRow(chain.failure);
        } catch (err) {
          if (disposed) return;
          if (err.status === 404) forgetTurn({ response: selected.latest, model: modelID });
          welcome.remove(); composer.setBusy(false); composer.setReadOnly(true);
          thread.errorRow(overgo.friendlyError(err));
          thread.node.appendChild(el('div', { class: 'row' },
            el('button', { class: 'btn alt', text: 'Retry loading', onclick: () => overgo.openConversation(selected) }),
            el('button', { class: 'btn alt', text: 'New conversation', onclick: () => overgo.openConversation(null) })));
          return;
        }
      }
      if (inflight && !disposed && !otherModel) {
        welcome.remove();
        const prompt = chain && (chain.messages || []).findLast(message => message.role === "user" && message.response === inflight);
        activeTurn = { response: inflight, model: modelID, previous: chain && chain.previous || "", assistant: thread.add("assistant", ""),
          prompt: prompt ? { text: prompt.content, attachments: [] } : null };
        resumeTurn(activeTurn);
      }
      if (!disposed) updateBusy();
      if (otherModel && !disposed) {
        composer.setReadOnly(true);
        thread.node.appendChild(el("div", { class: "note", role: "status" }, "This conversation belongs to another model. Choose its original model to continue, or start a new conversation. ",
          el("button", { class: "btn alt", text: "Choose model", onclick: () => document.getElementById("model-pill").click() }),
          el("button", { class: "btn alt", text: "New conversation", onclick: () => overgo.openConversation(null) })));
      }
      if (!disposed && focusedSetting && document.querySelector('#settings-dialog[open]') && document.activeElement === document.body) {
        settings.querySelector('[data-draft-setting="' + focusedSetting + '"]').focus();
      }
    },
  });
})();
