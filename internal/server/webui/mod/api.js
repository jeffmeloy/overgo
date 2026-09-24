/* API explorer: every declared route, rendered from the route table
   (/workspace/routes): a query or JSON editor, or the bound schema form,
   a send control through the shared API client, and the response. */
(function () {
  "use strict";
  const querySeparator = "?"; // a GET route's typed query follows its path
  window.overgo.registerTab({
    id: "api",
    async mount(panel, overgo) {
      const { el } = overgo;
      const table = await overgo.load((signal) => overgo.api.get("/workspace/routes", { signal }), { loading: "Loading the route table…" });
      if (!table) return;
      const count = el("div", { class: "note", text: table.routes.length + " routes from the server's route table" });
      const filter = el("input", { class: "text", type: "search", "aria-label": "Filter routes", placeholder: "Filter by path or method" });
      panel.append(filter, count);
      function routeCard(route) {
        const out = el("pre", { class: "mono scroll-box", hidden: true });
        const schema = (table.schemas || []).find((item) => item.route === route.path);
        const form = schema ? overgo.schemaForm(schema.schema, {}) : null;
        // GET and DELETE carry their inputs as a one-line query; every other method sends a JSON body.
        const queried = route.method === "GET" || route.method === "DELETE";
        const label = route.method + " " + route.path + (queried ? " query" : " body");
        const editor = form ? form.element : queried
          ? el("input", { class: "text", "aria-label": label, placeholder: "query, e.g. id=..." })
          : el("textarea", { class: "text", rows: "3", "aria-label": label, placeholder: "JSON body" });
        const send = el("button", { class: "btn alt", text: "send" });
        send.addEventListener("click", async () => {
          out.hidden = false;
          try {
            const value = form ? form.value() : editor.value.trim();
            let result;
            if (!queried) result = await overgo.api.post(route.path, form ? value : JSON.parse(value || "{}"));
            else {
              const query = form ? new URLSearchParams(value).toString() : value;
              const target = route.path + (query ? querySeparator + query : "");
              result = route.method === "GET" ? await overgo.api.get(target) : await overgo.api.delete(target);
            }
            out.textContent = typeof result === "string" ? result : JSON.stringify(result, null, 2);
          } catch (err) { out.textContent = overgo.friendlyError(err); }
        });
        const card = el("div", { class: "card" },
          el("div", { class: "row" }, el("span", { class: "tag", text: route.method }), el("span", { class: "mono", text: route.path }),
            el("span", { class: "note", text: route.authentication }), send),
          editor, out);
        card.dataset.route = (route.method + " " + route.path).toLowerCase();
        return card;
      }
      const cards = table.routes.map(routeCard);
      filter.addEventListener("input", () => {
        const needle = filter.value.trim().toLowerCase();
        let shown = 0;
        for (const card of cards) {
          card.hidden = needle !== "" && !card.dataset.route.includes(needle);
          if (!card.hidden) shown++;
        }
        count.textContent = needle ? shown + " of " + cards.length + " routes match" : cards.length + " routes from the server's route table";
      });
      panel.append(...cards);
    },
  });
})();
