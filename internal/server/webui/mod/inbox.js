(function () {
  "use strict";

  window.overgo.registerTab({
    id: "inbox",
    async mount(panel, overgo) {
      const { api, el, fmt } = overgo;
      const status = el("div", { class: "note" });
      const listHost = el("div");
      panel.replaceChildren(
        el("div", { class: "section-title", text: "Waiting on you" }), status, listHost);

      const artifactLink = overgo.artifactLink;

      // A decision rides the strip's binding path (operations_shell.js): it names the request advertised now.
      const decide = (item, action, answer) => overgo.act(status, async () => {
        await overgo.decideOperation(item.operation, action.code, answer);
        status.textContent = answer + " recorded for " + fmt.shortID(item.operation);
      });

      // decideSelected: the checked operations, each with its one action, decided together.
      const decideSelected = (answer) => overgo.act(status, async () => {
        const chosen = [...listHost.querySelectorAll("input[data-operation]:checked")];
        for (const box of chosen) await overgo.decideOperation(box.dataset.operation, box.dataset.action, answer);
        status.textContent = answer + " recorded for " + chosen.length + " operations";
      });
      function renderWaiting(waiting) {
        const selectable = waiting.filter((item) => (item.actions || []).length === 1);
        const bulk = selectable.length > 1 ? [el("div", { class: "row" },
          el("button", { class: "btn", text: "Grant selected", onclick: () => decideSelected("grant") }),
          el("button", { class: "btn alt", text: "Decline selected", onclick: () => decideSelected("decline") }))] : [];
        listHost.replaceChildren(...bulk, ...waiting.map((item) => renderItem(item, selectable.length > 1 && selectable.includes(item))));
      }

      function renderItem(item, selectable) {
        const actions = (item.actions || []).map((action) => el("div", { class: "row" },
          el("span", { class: "mono", text: action.summary || action.code }),
          el("button", { class: "btn", text: "Grant " + action.code, onclick: () => decide(item, action, "grant") }),
          el("button", { class: "btn alt", text: "Decline", onclick: () => decide(item, action, "decline") })));
        const prior = item.prior_decision ? el("div", { class: "note" }, "prior decision ", artifactLink(item.prior_decision.id),
          " / " + item.prior_decision.answer + " / " + item.prior_decision.tool) : null;
        const select = selectable ? el("input", { type: "checkbox", checked: true, "data-operation": item.operation, "data-action": item.actions[0].code,
          "aria-label": "Select " + item.task + " " + fmt.shortID(item.operation) }) : null;
        return el("div", { class: "card" },
          el("div", {}, ...(select ? [select, " "] : []), el("span", { class: "tag tag-danger", text: "waiting" }),
            " operation ", artifactLink(item.operation), " / " + item.task),
          el("div", { text: item.reason }),
          ...(prior ? [prior] : []),
          ...actions);
      }

      // Refreshes follow operation transitions, so the list stays in place while one reads.
      const refresh = overgo.read(listHost, async (signal) => (await api.get("/operations/inbox", { signal })).waiting || [],
        renderWaiting,
        { loading: null, empty: (waiting) => !waiting.length && "Nothing is waiting on an operator decision." });

      // Blocked operations announce themselves on the shared runtime stream; the inbox
      // refreshes after operation transitions, one read at a time with at most one queued.
      // It listens before its first read, so a transition during that read queues another.
      let reading = null, queued = false;
      function refreshSoon() {
        if (reading) { queued = true; return; }
        reading = refresh().finally(() => { reading = null; if (queued) { queued = false; refreshSoon(); } });
      }
      overgo.subscribe((event) => { if (event === "operation") refreshSoon(); });
      refreshSoon();
      await reading;
    },
  });
})();
