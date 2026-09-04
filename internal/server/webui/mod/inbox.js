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

      async function decide(item, action, answer) {
        try {
          await api.post("/operations/decision", {
            operation: item.operation, tool: action.code, answer,
          });
          status.textContent = answer + " recorded for " + fmt.shortID(item.operation);
          await refresh();
        } catch (err) {
          status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
        }
      }

      function renderItem(item) {
        const actions = (item.actions || []).map((action) => {
          const grant = el("button", { class: "btn", text: "Grant " + action.code });
          const decline = el("button", { class: "btn alt", text: "Decline" });
          grant.addEventListener("click", () => decide(item, action, "grant"));
          decline.addEventListener("click", () => decide(item, action, "decline"));
          return el("div", { class: "row" },
            el("span", { class: "mono", text: action.summary || action.code }), grant, decline);
        });
        const prior = item.prior_decision
          ? el("div", { class: "note" }, "prior decision ", artifactLink(item.prior_decision.id),
            " / " + item.prior_decision.answer + " / " + item.prior_decision.tool)
          : null;
        return el("div", { class: "card" },
          el("div", {}, el("span", { class: "tag tag-danger", text: "waiting" }),
            " operation ", artifactLink(item.operation), " / " + item.task),
          el("div", { text: item.reason }),
          ...(prior ? [prior] : []),
          ...actions);
      }

      async function refresh() {
        try {
          const inbox = await api.get("/operations/inbox");
          const waiting = inbox.waiting || [];
          if (!waiting.length) {
            listHost.replaceChildren(el("div", { class: "note", text: "Nothing is waiting on an operator decision." }));
            return;
          }
          listHost.replaceChildren(...waiting.map(renderItem));
        } catch (err) {
          status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
        }
      }

      await refresh();
      // Blocked operations announce themselves on the shared event
      // stream; the inbox refreshes on every operation transition rather
      // than polling.
      const stream = new AbortController();
      api.events("/agents/stream", (event) => {
        if (event === "operation") refresh();
      }, { signal: stream.signal }).catch((err) => {
        if (err.name !== "AbortError") status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
      });
      return () => stream.abort();
    },
  });
})();
