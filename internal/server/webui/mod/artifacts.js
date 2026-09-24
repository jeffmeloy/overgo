(function () {
  "use strict";
  window.overgo.registerTab({
    id: "artifacts",
    async mount(panel, overgo) {
      const { api, el, fmt } = overgo;
      const kind = el("input", { class: "text", placeholder: "kind", "aria-label": "Artifact kind" });
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

      function media(item) {
        if (!item.payload) return el("div", { class: "note", text: "payload unavailable" });
        const type = item.descriptor.media_type || "";
        const url = overgo.contentURL(item.descriptor.id);
        if (type.startsWith("image/")) return el("img", { src: url, alt: item.descriptor.id });
        if (type.startsWith("audio/")) return el("audio", { src: url, controls: true });
        if (type.startsWith("video/")) return el("video", { src: url, controls: true });
        return el("a", { href: url, text: "Download payload" });
      }
      // The newest load owns the gallery: a slower earlier answer does not replace it.
      const load = overgo.read(gallery, (signal) => {
        const query = new URLSearchParams();
        // A link elsewhere in the workbench shows one entry; Load lists the gallery again.
        const focus = overgo.focusedArtifact();
        if (focus) query.set("id", focus);
        else {
          if (cursor) query.set("cursor", cursor);
          if (kind.value.trim()) query.set("kind", kind.value.trim());
        }
        return api.get("/artifacts?" + query, { signal });
      }, (result) => {
        const focus = overgo.focusedArtifact();
        nextCursor = result.next || "";
        status.textContent = focus ? "showing " + fmt.shortID(focus) + "; Load lists every item" : result.artifacts.length + " of " + fmt.grouped(result.count) + " items";
        previous.disabled = !!focus || prior.length === 0;
        next.disabled = !!focus || !result.truncated;
        gallery.replaceChildren(...result.artifacts.map((item) => el("article", { class: "artifact" },
          media(item),
          el("div", { class: "mono", title: item.descriptor.id, text: fmt.shortID(item.descriptor.id) }),
          el("div", { class: "note", text: (item.descriptor.media_type || item.descriptor.id.split(":", 1)[0]) + " / " + fmt.bytes(item.descriptor.size) }),
          el("div", { class: "note", text: (item.producers || []).map(fmt.shortID).join(", ") || "producer unavailable" }))));
      }, { loading: "Loading artifacts…" });
      loadButton.addEventListener("click", () => overgo.openArtifact(null));
      next.addEventListener("click", () => { prior.push(cursor); cursor = nextCursor; load(); });
      previous.addEventListener("click", () => { cursor = prior.pop() || ""; load(); });
      const focused = () => { cursor = ""; prior = []; load(); };
      overgo.listen(window, "overgo-artifact-focus", focused);
      await load();
    },
  });
})();
