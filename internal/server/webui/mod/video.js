/* Video: native text-to-video through /v1/videos/generations, and
   reference-guided editing through /v1/videos/edits, over the shared
   composer and thread. Outputs are committed video artifacts played
   straight from the store as media cards; the edit surface takes its
   source video as a composer attachment. A model without the capability
   surfaces the server's typed refusal as an error row. */
(function () {
  "use strict";
  function generationSurface(id, options) {
    window.overgo.registerTab({
      id,
      mount(panel, overgo) {
        overgo.clear(panel);
        let controller = null;
        const thread = overgo.thread(panel);
        const composer = overgo.composer(panel, {
          placeholder: options.placeholder,
          sendLabel: options.sendLabel,
          accept: options.accept || [],
          multiple: false,
          attachLabel: options.attachLabel,
          onStop: () => { if (controller) controller.abort(); },
          onSubmit: async (text, attachments) => {
            if (controller) return;
            const body = { prompt: text };
            if (options.source) {
              if (!attachments.length) {
                thread.errorRow("pick a source video first");
                return;
              }
              body.source = attachments[0].dataURL;
            }
            controller = new AbortController();
            composer.setBusy(true);
            thread.add("user", text + (attachments.length ? "\n[video: " + attachments[0].name + "]" : ""));
            composer.clearInput();
            composer.clearAttachments();
            try {
              const result = await overgo.api.post(options.path, body, { signal: controller.signal });
              await thread.consume(overgo.streams.media("video", result, text));
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
  }
  generationSurface("video-gen", {
    path: "/v1/videos/generations", placeholder: "describe the video to generate", sendLabel: "generate",
  });
  generationSurface("video-edit", {
    path: "/v1/videos/edits", placeholder: "describe the edit to apply", sendLabel: "edit",
    kinds: ["video"], attachLabel: "source video", source: true,
  });
})();
