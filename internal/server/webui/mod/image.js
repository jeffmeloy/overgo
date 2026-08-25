/* Images: native text-to-image through /v1/images/generations. Outputs are
   artifact URLs, so the gallery renders straight from the store and each
   image links to its content. A model without an image-generation capability
   surfaces the server's typed refusal as the panel's empty state. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "image-gen",
    mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);

      const prompt = el("textarea", { class: "text", placeholder: "describe the image to generate" });
      const status = el("span", { class: "note" });
      const gallery = el("div", { class: "artifact-gallery" });
      const errors = el("div");
      const generate = el("button", { class: "btn" }, "generate");
      const cancel = el("button", { class: "btn alt", style: "display:none" }, "cancel");
      const run = overgo.runner(generate, cancel, {
        onCancel: () => { status.textContent = "cancelled"; },
        onError: (err) => {
          overgo.clear(errors);
          errors.appendChild(overgo.errorBanner(overgo.friendlyError(err)));
          status.textContent = "";
        },
      });

      generate.addEventListener("click", () => run(async (signal) => {
        overgo.clear(errors);
        status.textContent = "generating…";
        const body = { prompt: prompt.value };
        const result = await overgo.api.post("/v1/images/generations", body, { signal });
        status.textContent = (result.data || []).length + " image(s)";
        for (const item of result.data || []) {
          gallery.prepend(
            el("div", { class: "artifact" },
              el("a", { href: item.url, target: "_blank" }, el("img", { src: item.url, alt: prompt.value })),
              el("div", { class: "note", text: prompt.value })));
        }
      }));

      panel.append(
        prompt,
        el("div", { class: "chat-controls" }, generate, cancel, status),
        errors,
        gallery);
    },
  });
})();
