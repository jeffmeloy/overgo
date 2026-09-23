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
      async function decide(item, action, answer) {
        try {
          await overgo.decideOperation(item.operation, action.code, answer);
          status.textContent = answer + " recorded for " + fmt.shortID(item.operation);
          await refresh();
        } catch (err) { status.replaceChildren(overgo.failure(err)); }
      }

      function renderItem(item) {
        const actions = (item.actions || []).map((action) => el("div", { class: "row" },
          el("span", { class: "mono", text: action.summary || action.code }),
          el("button", { class: "btn", text: "Grant " + action.code, onclick: () => decide(item, action, "grant") }),
          el("button", { class: "btn alt", text: "Decline", onclick: () => decide(item, action, "decline") })));
        const prior = item.prior_decision ? el("div", { class: "note" }, "prior decision ", artifactLink(item.prior_decision.id),
          " / " + item.prior_decision.answer + " / " + item.prior_decision.tool) : null;
        return el("div", { class: "card" },
          el("div", {}, el("span", { class: "tag tag-danger", text: "waiting" }),
            " operation ", artifactLink(item.operation), " / " + item.task),
          el("div", { text: item.reason }),
          ...(prior ? [prior] : []),
          ...actions);
      }

      async function refresh() {
        try {
          const waiting = (await api.get("/operations/inbox")).waiting || [];
          listHost.replaceChildren(...(waiting.length ? waiting.map(renderItem) : [el("div", { class: "note", text: "Nothing is waiting on an operator decision." })]));
        } catch (err) { status.replaceChildren(overgo.failure(err)); }
      }

      // Blocked operations announce themselves on the shared runtime stream; the inbox
      // refreshes after operation transitions, one read at a time with at most one queued.
      // It listens before its first read, so a transition during that read queues another.
      let reading = null, queued = false;
      function refreshSoon() {
        if (reading) { queued = true; return; }
        reading = refresh().finally(() => { reading = null; if (queued) { queued = false; refreshSoon(); } });
      }
      const unsubscribe = overgo.runtimeEvents.subscribe((event) => { if (event === "operation") refreshSoon(); });
      refreshSoon();
      await reading;
      return unsubscribe;
    },
  });
})();
