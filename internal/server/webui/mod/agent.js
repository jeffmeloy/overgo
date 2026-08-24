/* Agent: drive one tool-using session over the served model. Presentation
   only -- every rule (unregistered refusal, inspection before mutation,
   exact mutation approval, the step bound) lives in the server's
   coordinator. This panel names a session, lists the registered tools with
   their effect badges, proposes a step, and appends the durable result to a
   running timeline. A mutation tool arms an explicit approval checkbox. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "agent",
    label: "Agent",
    section: "inference",
    mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);

      const session = el("input", { class: "text", value: "session-1", style: "width:180px" });
      const toolSelect = el("select", { class: "text", style: "width:220px" });
      const effectBadge = el("span", { class: "tag" });
      const args = el("textarea", { class: "text", rows: "3", placeholder: '{"arguments":"as strict JSON object"}' });
      const approve = el("input", { type: "checkbox" });
      const approveLabel = el("label", { class: "note", style: "display:none" }, approve, " approve this mutation");
      const propose = el("button", { class: "btn" }, "propose step");
      const note = el("div", { class: "note", text: "loading tools…" });
      const timeline = el("tbody");
      const manuals = {};

      function syncEffect() {
        const effect = manuals[toolSelect.value] || "";
        effectBadge.textContent = effect;
        effectBadge.className = "tag" + (effect === "mutation" ? " tag-danger" : "");
        const mutation = effect === "mutation";
        approveLabel.style.display = mutation ? "inline-flex" : "none";
        if (!mutation) approve.checked = false;
      }
      toolSelect.addEventListener("change", syncEffect);

      async function refreshTools() {
        try {
          const catalog = await overgo.api.get("/agent/tools");
          const tools = catalog.tools || [];
          clear(toolSelect);
          for (const tool of tools) {
            manuals[tool.name] = tool.effect || (tool.stale ? "stale" : "");
            toolSelect.appendChild(el("option", { value: tool.name, text: tool.name + (tool.stale ? " · stale" : "") }));
          }
          const coverage = catalog.coverage || {};
          note.textContent = tools.length
            ? "tools " + (coverage.published || tools.length) + "/" + (coverage.registered || tools.length)
            : "No registered tools — publish manuals with cmd/agent-tool.";
          syncEffect();
        } catch (err) {
          note.textContent = overgo.friendlyError(err);
        }
      }

      function appendStep(tool, effect, state, detail) {
        timeline.insertBefore(el("tr", null,
          el("td", { class: "mono", text: tool }),
          el("td", null, el("span", { class: "tag" + (effect === "mutation" ? " tag-danger" : ""), text: effect || "" })),
          el("td", null, el("span", { class: "tag" + (state === "refused" ? " tag-danger" : ""), text: state })),
          el("td", { class: "mono", text: (detail || "").slice(0, 400) })), timeline.firstChild);
      }

      propose.addEventListener("click", async () => {
        const tool = toolSelect.value;
        if (!tool) return;
        const effect = manuals[tool] || "";
        let parsed = {};
        const raw = args.value.trim();
        if (raw) {
          try { parsed = JSON.parse(raw); }
          catch (err) { note.textContent = "arguments must be valid JSON: " + err.message; return; }
        }
        propose.disabled = true;
        try {
          const body = await overgo.api.post("/agent/step", {
            session: session.value.trim(), tool, arguments: parsed, approve: approve.checked,
          });
          appendStep(tool, effect, "ok", JSON.stringify(body.result));
          note.textContent = "session " + body.session + " · step " + body.steps +
            (body.inspected ? " · inspected" : "");
        } catch (err) {
          appendStep(tool, effect, "refused", overgo.friendlyError(err));
          note.textContent = overgo.friendlyError(err);
        } finally {
          propose.disabled = false;
        }
      });

      panel.append(
        el("div", { class: "section-title", text: "Session" }),
        el("div", { class: "row" }, el("span", { class: "note", text: "session" }), session,
          toolSelect, effectBadge, approveLabel, propose),
        note,
        el("div", { class: "row" }, args),
        el("div", { class: "section-title", text: "Timeline" }),
        el("table", { class: "grid" },
          el("thead", null, el("tr", null,
            el("th", { text: "tool" }), el("th", { text: "effect" }), el("th", { text: "state" }), el("th", { text: "detail" }))),
          timeline));

      refreshTools();
    },
  });
})();
