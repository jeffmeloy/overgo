/* API explorer: every declared route, rendered from the route table
   (/workspace/routes): a query or JSON editor, or the bound schema form,
   a send control through the shared API client, and the response. */
(function () {
  "use strict";
  const querySeparator = "?"; // a GET route's typed query follows its path
  window.overgo.registerTab({
    id: "api",
    async mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);
      let table;
      try { table = await overgo.api.get("/workspace/routes"); } catch (err) { panel.appendChild(overgo.errorBanner(overgo.friendlyError(err))); return; }
      panel.append(el("div", { class: "note", text: table.routes.length + " routes from the server's route table" }));
      function routeCard(route) {
        const out = el("pre", { class: "mono scroll-box", hidden: true });
        const schema = (table.schemas || []).find((item) => item.route === route.path);
        const form = schema ? overgo.schemaForm(schema.schema, {}) : null;
        const editor = form ? form.element : el("textarea", { class: "text", rows: "3", placeholder: route.method === "GET" ? "query, e.g. id=..." : "JSON body" });
        const send = el("button", { class: "btn alt", text: "send" });
        send.addEventListener("click", async () => {
          out.hidden = false;
          try {
            const value = form ? form.value() : editor.value.trim();
            const query = value && !form ? querySeparator + value : "";
            const body = form ? value : JSON.parse(value || "{}");
            const result = route.method === "GET" ? await overgo.api.get(route.path + query) : await overgo.api.post(route.path, body);
            out.textContent = JSON.stringify(result, null, 2);
          } catch (err) { out.textContent = overgo.friendlyError(err); }
        });
        return el("div", { class: "card" },
          el("div", { class: "row" }, el("span", { class: "tag", text: route.method }), el("span", { class: "mono", text: route.path }),
            el("span", { class: "note", text: route.authentication }), send),
          editor, out);
      }
      panel.append(...table.routes.map(routeCard));
    },
  });
})();
