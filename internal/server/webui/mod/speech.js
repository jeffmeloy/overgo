/* Speech: native text-to-speech through /v1/audio/speech, over the shared
   composer and thread. The endpoint answers raw audio bytes, so each
   synthesis becomes a media card with a player over a local object URL. A
   model without a speech capability surfaces the server's typed refusal as
   an error row. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "speech",
    mount(panel, overgo) {
      const { clear } = overgo;
      clear(panel);
      let controller = null;
      const thread = overgo.thread(panel);
      const composer = overgo.composer(panel, {
        placeholder: "text to speak",
        sendLabel: "synthesize",
        onStop: () => { if (controller) controller.abort(); },
        onSubmit: async (text) => {
          if (controller) return;
          controller = new AbortController();
          composer.setBusy(true);
          thread.add("user", text);
          composer.clearInput();
          try {
            const response = await overgo.api.stream("/v1/audio/speech", { input: text }, { signal: controller.signal });
            await thread.consume(overgo.streams.blob("audio", await response.blob(), text));
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
