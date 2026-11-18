(function () {
  "use strict";
  // depthPreview: a depth section's map as its run makes it, as the MOSS Ultra workflow's PreviewImage shows it, before
  // any run; offered while the page holds a depth control. request() is the run's task, recipe and input. The estimate
  // runs once per press, never again on its own.
  function depthPreview(host, request, overgo) {
    const { el, api } = overgo;
    const shown = el("div", { class: "depth-preview" });
    const open = el("button", { class: "btn alt", type: "button", text: "Preview depth map" });
    open.addEventListener("click", async () => {
      open.disabled = true;
      shown.replaceChildren(el("div", { class: "note", role: "status", text: "Estimating depth…" }));
      try {
        const preview = await api.post("/generation/preview", { ...request(), position: 0, limit: 1 });
        shown.replaceChildren(el("img", { src: overgo.contentURL(preview.output), alt: "Depth map", class: "depth-preview-image" }));
      } catch (err) { shown.replaceChildren(overgo.failure(err)); } finally { open.disabled = false; }
    });
    host.replaceChildren(el("div", { class: "row" }, open), shown);
    const panel = host.parentElement;
    const offer = () => { host.hidden = !panel.querySelector('fieldset[data-control="control"]'); };
    new MutationObserver(offer).observe(panel, { childList: true, subtree: true });
    offer();
  }
  window.overgo.workflowWorkspace({ id: "generation", scope: "generation", preview: depthPreview });
})();
