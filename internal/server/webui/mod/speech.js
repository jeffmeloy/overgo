/* Speech: native text-to-speech through /v1/audio/speech. The endpoint
   answers raw audio bytes, so each synthesis becomes a player over a local
   object URL, newest first. A model without a speech capability surfaces the
   server's typed refusal as the panel's empty state. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "speech",
    label: "Speech",
    section: "inference",
    mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);

      const input = el("textarea", { class: "text", placeholder: "text to speak" });
      const status = el("span", { class: "note" });
      const takes = el("div");
      const errors = el("div");
      const speak = el("button", { class: "btn" }, "synthesize");
      const cancel = el("button", { class: "btn alt", style: "display:none" }, "cancel");
      const run = overgo.runner(speak, cancel, {
        onCancel: () => { status.textContent = "cancelled"; },
        onError: (err) => {
          overgo.clear(errors);
          errors.appendChild(overgo.errorBanner(overgo.friendlyError(err)));
          status.textContent = "";
        },
      });

      speak.addEventListener("click", () => run(async (signal) => {
        overgo.clear(errors);
        status.textContent = "synthesizing…";
        const response = await overgo.api.stream("/v1/audio/speech", { input: input.value }, { signal });
        const blob = await response.blob();
        const audio = el("audio", { controls: "", src: URL.createObjectURL(blob) });
        takes.prepend(el("div", { class: "artifact" }, audio,
          el("div", { class: "preview-text", text: input.value })));
        status.textContent = blob.type + " · " + overgo.fmt.bytes(blob.size);
      }));

      panel.append(
        input,
        el("div", { class: "chat-controls" }, speak, cancel, status),
        errors,
        takes);
    },
  });
})();
