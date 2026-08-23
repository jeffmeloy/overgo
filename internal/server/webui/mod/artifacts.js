(function () {
  "use strict";
  window.overgo.registerTab({
    id: "artifacts",
    label: "Artifacts",
    section: "workbench",
    async mount(panel, overgo) {
      const { api, el, clear, fmt } = overgo;
      clear(panel);
      const kind = el("input", { class: "text", placeholder: "kind" });
      const loadButton = el("button", { class: "btn", text: "Load" });
      const previous = el("button", { class: "btn alt", text: "Previous" });
      const next = el("button", { class: "btn alt", text: "Next" });
      const status = el("span", { class: "note" });
      const gallery = el("div", { class: "artifact-gallery" });
      panel.append(
        el("div", { class: "section-title", text: "Artifacts" }),
        el("div", { class: "row" }, kind, loadButton, previous, next, status),
        gallery);
      let cursor = "";
      let nextCursor = "";
      let prior = [];

      function contentURL(id) { return "/artifacts/content?id=" + encodeURIComponent(id); }
      function media(item) {
        if (!item.payload) return el("div", { class: "note", text: "payload unavailable" });
        const type = item.descriptor.media_type || "";
        const url = contentURL(item.descriptor.id);
        if (type.startsWith("image/")) return el("img", { src: url, alt: item.descriptor.id });
        if (type.startsWith("audio/")) return el("audio", { src: url, controls: true });
        if (type.startsWith("video/")) return el("video", { src: url, controls: true });
        return el("a", { href: url, text: "Open payload" });
      }
      async function load() {
        const query = new URLSearchParams();
        if (cursor) query.set("cursor", cursor);
        if (kind.value.trim()) query.set("kind", kind.value.trim());
        try {
          const result = await api.get("/artifacts?" + query);
          nextCursor = result.next || "";
          status.textContent = result.artifacts.length + " of " + fmt.grouped(result.count) + " items";
          previous.disabled = prior.length === 0;
          next.disabled = !result.truncated;
          gallery.replaceChildren(...result.artifacts.map((item) => el("article", { class: "artifact" },
            media(item),
            el("div", { class: "mono", title: item.descriptor.id, text: fmt.shortID(item.descriptor.id) }),
            el("div", { class: "note", text: (item.descriptor.media_type || item.descriptor.id.split(":", 1)[0]) + " / " + fmt.bytes(item.descriptor.size) }),
            el("div", { class: "note", text: (item.producers || []).map(fmt.shortID).join(", ") || "producer unavailable" }))));
        } catch (err) {
          gallery.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
        }
      }
      loadButton.addEventListener("click", () => { cursor = ""; prior = []; load(); });
      next.addEventListener("click", () => { prior.push(cursor); cursor = nextCursor; load(); });
      previous.addEventListener("click", () => { cursor = prior.pop() || ""; load(); });
      await load();
    },
  });
})();
