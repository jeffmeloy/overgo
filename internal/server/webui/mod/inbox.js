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
        } catch (err) { status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
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
        } catch (err) { status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
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
