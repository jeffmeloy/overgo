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
      // The page leads with the task: pick an agent and chat. Creating
      // one is a fold with three plain fields; every operator panel --
      // manual tool steps, the full definition editor, retrieval,
      // automations, observables -- lives behind Advanced folds.
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
      // One session per page visit, named automatically: nobody types a
      // session identity to talk to an agent. The Advanced panel can
      // still override it.
      const autoSession = "chat-" + new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");

      // The creation surface is everything a person needs: a name,
      // plain instructions, and tool checkboxes. Every identity -- the
      // prompt artifact, the served recipe, the exact manuals, the
      // default policy -- derives server-side at /agents/create.
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

      function activeAgent() {
        return inventory.find((item) => item.name === selected);
      }

      function showError(err) {
        status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
      }

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
        const table = el("table", { class: "grid" }, overgo.headerRow(["name", "definition", "state", "authority", "select"]));
        for (const item of inventory) {
          const choose = el("button", { class: "btn alt", text: item.name === selected ? "Selected" : "Open" });
          choose.addEventListener("click", () => { selected = item.name; renderAll(); });
          table.append(overgo.tableRow([item.name, artifactLink(item.definition), el("span", { text: item.refusal || item.state }), artifactLink(item.activation), choose]));
        }
        const pause = el("button", { class: "btn alt", text: "Pause", disabled: !selected });
        const resume = el("button", { class: "btn", text: "Resume", disabled: !selected });
        pause.addEventListener("click", () => transition("paused"));
        resume.addEventListener("click", () => transition("active"));
        inventoryHost.replaceChildren(table, el("div", { class: "row" }, pause, resume));
      }

      publish.addEventListener("click", async () => {
        if (!definitionForm.validate()) {
          status.textContent = "Complete every required definition field";
          return;
        }
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

      // The conversation is the shared thread and composer (composer.js):
      // the agent's replies arrive complete through /agents/chat and render
      // through the same event vocabulary the streamed surfaces use, so
      // attachments, stop, markdown and copy behave here as in Chat.
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
                ? { role: "user", content: [{ type: "text", text: content }, ...parts] }
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
              } catch (err) {
                chatThread.errorRow(err.name === "AbortError" ? "stopped" : overgo.friendlyError(err));
              } finally {
                chatController = null;
                const current = activeAgent();
                chatComposer.setBusy(!current || current.state !== "active");
              }
            },
          });
        }
        chatComposer.setBusy(!agent || agent.state !== "active");
      }

      function renderTools() {
        const agent = activeAgent();
        const allowed = new Set((agent && agent.tools) || []);
        const select = el("select", { class: "text" });
        for (const tool of tools.filter((item) => allowed.has(item.manual))) {
          select.append(el("option", { value: tool.name, text: tool.name + " / " + tool.effect }));
        }
        const session = el("input", { class: "text", placeholder: "session identity", value: autoSession });
        const args = el("textarea", { class: "text", rows: "2", placeholder: "Strict JSON arguments" });
        const decisionHost = el("div");
        const review = el("button", { class: "btn alt", text: "Review decision", disabled: !agent });
        const execute = el("button", { class: "btn alt", text: "Execute inspection", disabled: !agent });
        const approve = el("button", { class: "btn", text: "Approve and execute", disabled: true });

        // The decision surface: what a grant would BIND -- the manual's
        // immutable identity and the exact argument bytes -- beside any
        // committed decision, with the approved-vs-proposed diff when the
        // committed decision binds different facts.
        function renderDecision(preview) {
          const effectTag = el("span", {
            class: preview.effect === "mutation" ? "tag tag-danger" : "tag", text: preview.effect,
          });
          const rows = [
            el("div", {}, effectTag, " ", preview.tool, " / manual ", artifactLink(preview.manual)),
            el("div", { class: "mono", text: "grant binds: " + preview.arguments.join(" ") }),
          ];
          if (!preview.decision) {
            rows.push(el("div", { class: "note", text: preview.effect === "mutation"
              ? "No committed decision yet: approving records a durable grant for exactly these bytes."
              : "Inspections are effect-free and need no decision." }));
          } else if (preview.binds) {
            rows.push(el("div", {}, el("span", { class: "tag", text: preview.decision.answer }),
              " committed decision ", artifactLink(preview.decision.id), " binds these exact facts"));
          } else {
            rows.push(el("div", {}, el("span", { class: "tag tag-danger", text: "does not bind" }),
              " committed decision ", artifactLink(preview.decision.id), " (" + preview.decision.answer + ")"),
              el("div", { class: "mono", text: "approved: " + preview.decision.arguments.join(" ") }),
              el("div", { class: "mono", text: "proposed: " + preview.arguments.join(" ") }));
          }
          decisionHost.replaceChildren(el("div", { class: "card" }, ...rows));
          approve.disabled = preview.effect !== "mutation";
          execute.disabled = preview.effect === "mutation";
        }

        async function preview() {
          const body = {
            agent: selected, session: session.value.trim(), tool: select.value,
            arguments: JSON.parse(args.value || "{}"),
          };
          return api.post("/agents/approval", body);
        }

        review.addEventListener("click", async () => {
          try { renderDecision(await preview()); } catch (err) { showError(err); }
        });

        async function step(approveFlag) {
          try {
            const result = await api.post("/agents/step", {
              agent: selected, session: session.value.trim(), tool: select.value,
              arguments: JSON.parse(args.value || "{}"), approve: approveFlag,
            });
            status.textContent = "tool step " + result.steps + " / " + fmt.shortID(result.interaction);
            decisionHost.replaceChildren();
            await renderEvidence(session.value.trim());
          } catch (err) {
            showError(err);
            // A binding refusal re-renders the diff so the operator sees
            // what the committed decision actually covers.
            try { renderDecision(await preview()); } catch (previewErr) { void previewErr; }
          }
        }
        execute.addEventListener("click", () => step(false));
        approve.addEventListener("click", () => step(true));

        const sessionListHost = el("div");
        const listSessions = el("button", { class: "btn alt", text: "Sessions" });
        // The session list is the ledger's memory: every durable session
        // with its recorded steps against the bound and its inspection
        // state, resumable by one click into the session box.
        listSessions.addEventListener("click", async () => {
          try {
            const listing = await api.get("/agent/sessions");
            sessionListHost.replaceChildren(...(listing.sessions || []).map((item) => {
              const resume = el("button", { class: "btn alt", text: "Resume" });
              resume.addEventListener("click", () => {
                session.value = item.id.includes(":") ? item.id.split(":").pop() : item.id;
                renderEvidence(session.value);
              });
              return el("div", { class: "card" },
                el("span", { class: "mono", text: item.id }),
                " steps " + item.steps + (item.inspected ? " / inspected " : " / uninspected "),
                artifactLink(item.interaction), " ", resume);
            }));
          } catch (err) { showError(err); }
        });
        toolsHost.replaceChildren(
          el("div", { class: "row" }, session, select, review, execute, approve, listSessions),
          args, decisionHost, sessionListHost);
      }

      function renderRetrieval() {
        const projection = el("input", { class: "text", placeholder: "retrieval projection identity" });
        const policy = el("input", { class: "text", placeholder: "rerank policy identity" });
        const query = el("input", { class: "text", placeholder: "retrieval query" });
        const limit = el("input", { class: "text", type: "number", min: "1", placeholder: "result limit" });
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
        const select = el("select", { class: "text" });
        for (const item of (agent && agent.automations) || []) {
          select.append(el("option", { value: item.id, text: item.name }));
        }
        const key = el("input", { class: "text", placeholder: "idempotency key" });
        const destination = el("input", { class: "text", placeholder: "approved destination" });
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

      // renderProvenance expands one step's full evidence walk: the
      // interaction, the exact manual identity, the receipt chain from
      // completed back to admitted, the committed decision, and the
      // result -- every link an artifact the operator can open.
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
          if (walk.result) {
            parts.push(el("div", { class: "mono", text: "result " + (walk.result.error ? "ERROR " : "") + walk.result.content }));
          }
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

      function renderAll() {
        renderInventory(); renderCreate(); renderChat(); renderTools(); renderRetrieval(); renderAutomations();
      }

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
