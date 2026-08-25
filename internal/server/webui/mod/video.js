/* Video: native text-to-video through /v1/videos/generations, and
   reference-guided editing through /v1/videos/edits. Outputs are
   committed video artifacts played straight from the store. A model
   without the capability surfaces the server's typed refusal. */
(function () {
  "use strict";

  function videoCard(el, url, caption) {
    return el("div", { class: "artifact" },
      el("video", { src: url, controls: "", style: "max-width:420px" }),
      el("div", { class: "note", text: caption }));
  }

  window.overgo.registerTab({
    id: "video-gen",
    mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);
      const prompt = el("textarea", { class: "text", placeholder: "describe the video to generate" });
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
        const result = await overgo.api.post("/v1/videos/generations", { prompt: prompt.value }, { signal });
        status.textContent = (result.data || []).length + " video(s)";
        for (const item of result.data || []) gallery.prepend(videoCard(el, item.url, prompt.value));
      }));
      panel.append(prompt, el("div", { class: "chat-controls" }, generate, cancel, status), errors, gallery);
    },
  });

  window.overgo.registerTab({
    id: "video-edit",
    mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);
      const prompt = el("textarea", { class: "text", placeholder: "describe the edit to apply" });
      const status = el("span", { class: "note" });
      const gallery = el("div", { class: "artifact-gallery" });
      const errors = el("div");
      const sourceHost = el("div", { class: "row" });
      let sourceDataURL = "";
      const picker = el("input", { type: "file", style: "display:none", accept: "video/mp4" });
      const pick = el("button", { class: "btn alt", onclick: () => picker.click() }, "source video");
      picker.addEventListener("change", () => {
        const file = picker.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          sourceDataURL = reader.result;
          sourceHost.replaceChildren(el("video", { src: sourceDataURL, controls: "", style: "max-width:320px" }),
            el("span", { class: "note", text: file.name }));
        };
        reader.readAsDataURL(file);
        picker.value = "";
      });
      const edit = el("button", { class: "btn" }, "edit");
      const cancel = el("button", { class: "btn alt", style: "display:none" }, "cancel");
      const run = overgo.runner(edit, cancel, {
        onCancel: () => { status.textContent = "cancelled"; },
        onError: (err) => {
          overgo.clear(errors);
          errors.appendChild(overgo.errorBanner(overgo.friendlyError(err)));
          status.textContent = "";
        },
      });
      edit.addEventListener("click", () => run(async (signal) => {
        overgo.clear(errors);
        if (!sourceDataURL) {
          errors.appendChild(overgo.errorBanner("pick a source video first"));
          return;
        }
        status.textContent = "editing…";
        const result = await overgo.api.post("/v1/videos/edits",
          { prompt: prompt.value, source: sourceDataURL }, { signal });
        status.textContent = (result.data || []).length + " edited video(s)";
        for (const item of result.data || []) gallery.prepend(videoCard(el, item.url, prompt.value));
      }));
      panel.append(prompt, el("div", { class: "chat-controls" }, pick, picker, edit, cancel, status),
        sourceHost, errors, gallery);
    },
  });
})();
