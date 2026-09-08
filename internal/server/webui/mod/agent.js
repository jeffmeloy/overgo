(function () {
  "use strict";

  window.overgo.registerTab({
    id: "agent",
    async mount(panel, overgo) {
      const { api, el, fmt } = overgo;
      const status = el("div", { class: "note" });
      const createHost = el("div");
      const inventoryHost = el("div");
      const editorHost = el("div");
      const conversationHost = el("div");
      const toolsHost = el("div");
      const retrievalHost = el("div");
      const automationHost = el("div");
      const evidenceHost = el("div");
      // The page leads with the task: pick an agent and chat; creating one is a fold with three plain
      // fields, and every operator panel lives behind Advanced folds.
      panel.replaceChildren(
        el("div", { class: "section-title", text: "Agents" }), status,
        overgo.fold("Your agents", true, inventoryHost),
        overgo.fold("New agent", false, createHost),
        overgo.fold("Chat", true, conversationHost),
        overgo.fold("Advanced: manual tool steps and approvals", false, toolsHost),
        overgo.fold("Advanced: full definition editor", false, editorHost),
        overgo.fold("Advanced: retrieval evidence", false, retrievalHost),
        overgo.fold("Advanced: attached automations", false, automationHost),
        overgo.fold("Advanced: structured observables", false, evidenceHost));

      const schema = await api.get("/workspace/schema?id=agent-definition");
      const definitionForm = overgo.schemaForm(schema, {});
      const publish = el("button", { class: "btn", text: "Publish and activate" });
      editorHost.append(definitionForm.element, publish);

      let inventory = [];
      let selected = "";
      let tools = [];
      const sessions = new Map();
      // One session per page visit, named automatically; the Advanced panel can still override it.
      const autoSession = "chat-" + new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");

      // The creation surface: a name, plain instructions, tool checkboxes; every identity derives at /agents/create.
      function renderCreate() {
        const name = el("input", { class: "text", placeholder: "agent name (lowercase)" });
        const instructions = el("textarea", { class: "text", rows: "3", placeholder: "what should this agent do?" });
        const boxes = tools.map((tool) => {
          const box = el("input", { type: "checkbox" });
          return { tool, box, row: el("label", { class: "note" }, box, " " + tool.name + " / " + tool.effect) };
        });
        const create = el("button", { class: "btn", text: "Create agent" });
        const note = el("span", { class: "note" });
        create.addEventListener("click", async () => {
          try {
            const chosen = boxes.filter((item) => item.box.checked).map((item) => item.tool.name);
            await api.post("/agents/create", {
              name: name.value.trim(), instructions: instructions.value.trim(), tools: chosen,
            });
            selected = name.value.trim();
            inventory = await api.get("/agents");
            note.textContent = "created and activated";
            renderAll();
          } catch (err) { showError(err); }
        });
        createHost.replaceChildren(name, instructions,
          el("div", { class: "row" }, ...boxes.map((item) => item.row)),
          el("div", { class: "row" }, create, note));
      }

      function activeAgent() { return inventory.find((item) => item.name === selected); }

      const showError = overgo.reporter(status);

      const artifactLink = overgo.artifactLink;

      function openOperation(id) {
        history.pushState({}, "", "?operation=" + encodeURIComponent(id));
        window.dispatchEvent(new PopStateEvent("popstate"));
      }

      async function transition(state) {
        if (!selected) return;
        try {
          const active = await api.post("/agents/state", { name: selected, state });
          status.textContent = state + " / " + fmt.shortID(active.activation && active.activation.authority);
          inventory = await api.get("/agents");
          renderAll();
        } catch (err) { showError(err); }
      }

      function renderInventory() {
        if (!selected && inventory.length) selected = inventory[0].name;
        if (selected && !inventory.some((item) => item.name === selected)) selected = "";
        const table = overgo.table(["name", "definition", "state", "authority", "select"], inventory.map((item) => [
          item.name, artifactLink(item.definition), el("span", { text: item.refusal || item.state }), artifactLink(item.activation),
          el("button", { class: "btn alt", text: item.name === selected ? "Selected" : "Open", onclick: () => { selected = item.name; renderAll(); } })]));
        const pause = el("button", { class: "btn alt", text: "Pause", disabled: !selected });
        const resume = el("button", { class: "btn", text: "Resume", disabled: !selected });
        pause.addEventListener("click", () => transition("paused"));
        resume.addEventListener("click", () => transition("active"));
        inventoryHost.replaceChildren(table, el("div", { class: "row" }, pause, resume));
      }

      publish.addEventListener("click", async () => {
        if (!definitionForm.validate()) { status.textContent = "Complete every required definition field"; return; }
        try {
          const created = await api.post("/agents/definitions", definitionForm.value());
          await api.post("/agents/activate", { definition: created.id });
          definitionForm.markSaved();
          selected = created.definition.name;
          status.textContent = "activated / " + fmt.shortID(created.id);
          inventory = await api.get("/agents");
          renderAll();
        } catch (err) { showError(err); }
      });

      // The conversation is the shared thread and composer: replies arrive complete through /agents/chat
      // and render through the same event vocabulary, so attachments, stop, markdown and copy match Chat.
      let chatThread = null;
      let chatComposer = null;
      let chatController = null;
      function renderChat() {
        const agent = activeAgent();
        if (!chatThread) {
          chatThread = overgo.thread(conversationHost);
          chatComposer = overgo.composer(conversationHost, {
            placeholder: "Message the selected agent",
            sendLabel: "Send",
            onStop: () => { if (chatController) chatController.abort(); },
            onSubmit: async (content, attachments) => {
              if (chatController) return;
              const parts = chatComposer.attachmentParts();
              const history = (sessions.get(selected) || []).slice();
              history.push(parts.length
                ? { role: "user", content: [...parts, { type: "text", text: content }] }
                : { role: "user", content });
              chatThread.add("user", overgo.userLine(content, attachments));
              chatComposer.clearInput();
              chatComposer.clearAttachments();
              chatController = new AbortController();
              chatComposer.setBusy(true);
              try {
                const result = await api.post("/agents/chat", { agent: selected, messages: history }, { signal: chatController.signal });
                const assistant = chatThread.add("assistant", "");
                await chatThread.consume(overgo.streams.reply(result), assistant);
                if (assistant.content) history.push({ role: "assistant", content: assistant.content });
                sessions.set(selected, history);
              } catch (err) { chatThread.errorRow(err.name === "AbortError" ? "stopped" : overgo.friendlyError(err)); } finally {
                chatController = null;
                const current = activeAgent();
                chatComposer.setBusy(!current || current.state !== "active");
              }
            },
          });
        }
        chatComposer.setBusy(!agent || agent.state !== "active");
      }

      // The manual tool-step surface is the shared one; this tab adds the session box and session list.
      let toolSurface = null;
      function renderTools() {
        if (!toolSurface) {
          const session = el("input", { class: "text", placeholder: "session identity", value: autoSession });
          const sessionListHost = el("div");
          const listSessions = el("button", { class: "btn alt", text: "Sessions" });
          // The session list: every durable session with its steps against the bound, resumable by one click.
          listSessions.addEventListener("click", async () => {
            try {
              const listing = await api.get("/agent/sessions");
              sessionListHost.replaceChildren(...(listing.sessions || []).map((item) => {
                const resume = el("button", { class: "btn alt", text: "Resume" });
                resume.addEventListener("click", () => { session.value = item.id.includes(":") ? item.id.split(":").pop() : item.id; renderEvidence(session.value); });
                return el("div", { class: "card" },
                  el("span", { class: "mono", text: item.id }),
                  " steps " + item.steps + " / " + item.bound + (item.inspected ? " / inspected " : " / uninspected "),
                  artifactLink(item.interaction), " ", resume);
              }));
            } catch (err) { showError(err); }
          });
          toolSurface = overgo.toolStep(toolsHost, {
            agent: () => selected, session: () => session.value.trim(), thread: () => chatThread,
            controls: [session, listSessions], onError: showError,
            onStep: (result) => { status.textContent = "tool step " + result.steps + " / " + fmt.shortID(result.interaction); renderEvidence(session.value.trim()); },
          });
          toolsHost.appendChild(sessionListHost);
        }
        toolSurface.setAgent(activeAgent(), tools);
      }

      function renderRetrieval() {
        const [projection, policy, query, limit] = ["retrieval projection identity", "rerank policy identity", "retrieval query", "result limit"]
          .map((placeholder) => el("input", { class: "text", placeholder }));
        const results = el("div");
        const search = el("button", { class: "btn", text: "Search", disabled: !selected });
        search.addEventListener("click", async () => {
          try {
            const found = await api.post("/agents/retrieval", {
              agent: selected, projection: projection.value.trim(), rerank_policy: policy.value.trim(),
              query: query.value, limit: Number(limit.value),
            });
            results.replaceChildren(...found.map((item) => el("div", { class: "card" },
              artifactLink(item.citation.source), " / ", artifactLink(item.citation.chunk),
              el("div", { text: item.citation.text }))));
          } catch (err) { showError(err); }
        });
        retrievalHost.replaceChildren(el("div", { class: "row" }, projection, policy, query, limit, search), results);
      }

      function renderAutomations() {
        const agent = activeAgent();
        const select = el("select", { class: "text", "aria-label": "automation" }, ...((agent && agent.automations) || []).map((item) => el("option", { value: item.id, text: item.name })));
        const [key, destination] = ["idempotency key", "approved destination"].map((placeholder) => el("input", { class: "text", placeholder }));
        const inputs = el("textarea", { class: "text", rows: "2", placeholder: "Strict JSON inputs" });
        const run = el("button", { class: "btn", text: "Run attachment", disabled: !select.value });
        run.addEventListener("click", async () => {
          try {
            const execution = await api.post("/agents/automation", {
              agent: selected, automation: select.value, key: key.value.trim(),
              destination: destination.value.trim(), inputs: JSON.parse(inputs.value || "{}"),
            });
            status.textContent = "automation admitted / " + fmt.shortID(execution.operation);
            openOperation(execution.operation);
          } catch (err) { showError(err); }
        });
        automationHost.replaceChildren(el("div", { class: "row" }, select, key, destination, run), inputs);
      }

      // renderProvenance expands one step's evidence walk: interaction, manual, receipts, decision, result.
      async function renderProvenance(host, session, step) {
        try {
          const walk = await api.get("/agent/provenance?session=" + encodeURIComponent(selected + ":" + session) + "&step=" + step);
          const parts = [
            el("div", {}, "interaction ", artifactLink(walk.interaction), " / transcript ", artifactLink(walk.transcript)),
            el("div", {}, "tool " + walk.tool + " / manual ", walk.manual ? artifactLink(walk.manual) : el("span", { text: "by name" })),
            el("div", { class: "mono", text: "arguments " + (walk.arguments || "{}") }),
          ];
          if (walk.receipts) {
            parts.push(el("div", {}, "receipts ", ...walk.receipts.flatMap((receipt) => [
              el("span", { class: receipt.state === "failed" ? "tag tag-danger" : "tag", text: receipt.state }), " ",
              artifactLink(receipt.id), " ",
            ])));
          }
          if (walk.decision) {
            parts.push(el("div", {}, "decision ", artifactLink(walk.decision.id),
              " / " + walk.decision.answer + " for " + walk.decision.tool));
          }
          if (walk.result) parts.push(el("div", { class: "mono", text: "result " + (walk.result.error ? "ERROR " : "") + walk.result.content }));
          host.replaceChildren(el("div", { class: "card" }, ...parts));
        } catch (err) { showError(err); }
      }

      async function renderEvidence(session) {
        if (!selected || !session) {
          evidenceHost.replaceChildren(el("div", { class: "note", text: "Run a tool step to load its durable observables." }));
          return;
        }
        try {
          const rows = await api.get("/agents/evidence?session=" + encodeURIComponent(selected + ":" + session));
          evidenceHost.replaceChildren(...rows.map((item) => {
            const provenanceHost = el("div");
            const open = el("button", { class: "btn alt", text: "Provenance" });
            open.addEventListener("click", () => renderProvenance(provenanceHost, session, item.step));
            return el("div", { class: "card" },
              "step " + item.step + " / ", artifactLink(item.interaction), " / transcript ", artifactLink(item.transcript), " ", open,
              ...((item.calls || []).map((call) => el("div", { text: "call " + call.name + " / " + fmt.shortID(call.manual) }))),
              ...((item.results || []).map((result) => el("div", { text: "result " + result.tool_call_id + (result.error ? " / error" : " / complete") }))),
              provenanceHost);
          }));
        } catch (err) { showError(err); }
      }

      function renderAll() { renderInventory(); renderCreate(); renderChat(); renderTools(); renderRetrieval(); renderAutomations(); }

      tools = (await api.get("/agent/tools")).tools || [];
      inventory = await api.get("/agents");
      renderAll();
      const stream = new AbortController();
      api.events("/agents/stream", (event, data) => {
        if (event === "agent.inventory") { inventory = data; renderAll(); }
        if (event === "operation" && data.status && data.status.id) status.textContent =
          "operation " + fmt.shortID(data.status.id) + " / " + data.status.state;
      }, { signal: stream.signal }).catch((err) => {
        if (err.name !== "AbortError") showError(err);
      });
      return () => { stream.abort(); definitionForm.dispose(); };
    },
  });
})();
