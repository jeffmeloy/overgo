/* Images: native text-to-image through /v1/images/generations, over the
   shared composer and thread. Outputs are artifact URLs, so each image
   renders straight from the store as a media card and links to its
   content. A model without an image-generation capability surfaces the
   server's typed refusal as an error row. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "image-gen",
    mount(panel, overgo) {
      const { clear } = overgo;
      clear(panel);
      let controller = null;
      const thread = overgo.thread(panel);
      const composer = overgo.composer(panel, {
        placeholder: "describe the image to generate",
        sendLabel: "generate",
        onStop: () => { if (controller) controller.abort(); },
        onSubmit: async (text) => {
          if (controller) return;
          controller = new AbortController();
          composer.setBusy(true);
          thread.add("user", text);
          composer.clearInput();
          try {
            const result = await overgo.api.post("/v1/images/generations", { prompt: text }, { signal: controller.signal });
            await thread.consume(overgo.streams.media("image", result, text));
          } catch (err) {
            thread.errorRow(err.name === "AbortError" ? "cancelled" : overgo.friendlyError(err));
          } finally {
            controller = null;
            composer.setBusy(false);
          }
        },
      });
    },
  });
})();
