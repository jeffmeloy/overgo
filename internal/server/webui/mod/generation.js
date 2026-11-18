(function () {
  "use strict";
  // The generation tab is MOSS's workbench as it is every media model's page: a model's controls and presets, a depth
  // section's preview before its run, and a run's pictures shown and compared.
  let request = null; // the run's task, recipe and input as the page composes them, once the tab is mounted

  // depthPreview: a depth section's map as its run makes it, as the MOSS Ultra workflow's PreviewImage shows it, before
  // any run; offered while the page holds a depth control. The estimate runs once per press, never again on its own.
  function depthPreview(host, composed, overgo) {
    const { el, api } = overgo;
    request = composed;
    const shown = el("div", { class: "depth-preview" });
    const open = el("button", { class: "btn alt", type: "button", text: "Preview depth map" });
    open.addEventListener("click", async () => {
      open.disabled = true;
      shown.replaceChildren(el("div", { class: "note", role: "status", text: "Estimating depth…" }));
      try {
        const preview = await api.post("/generation/preview", { ...composed(), position: 0, limit: 1 });
        shown.replaceChildren(el("img", { src: overgo.contentURL(preview.output), alt: "Depth map", class: "depth-preview-image" }));
      } catch (err) { shown.replaceChildren(overgo.failure(err)); } finally { open.disabled = false; }
    });
    host.replaceChildren(el("div", { class: "row" }, open), shown);
    const panel = host.parentElement;
    const offer = () => { host.hidden = !panel.querySelector('fieldset[data-control="control"]'); };
    new MutationObserver(offer).observe(panel, { childList: true, subtree: true });
    offer();
  }

  // pictures are the pictures this tab's runs drew and the pictures their requests started from, newest first, each
  // once, which the A/B compare offers.
  const pictures = [];
  function remember(id, label) {
    const held = pictures.findIndex((picture) => picture.id === id);
    if (held >= 0) pictures.splice(held, 1);
    pictures.unshift({ id, label });
  }

  // abCompare: two pictures, B over A to a divide a slider moves, as the MOSS Ultra workflow's image comparers show a
  // section's picture against the one it started from (rgthree's Image Comparer, nodes 611 and 669).
  function abCompare(overgo, a, b) {
    const { el, contentURL } = overgo;
    const choose = (label, chosen) => el("select", { class: "text w-auto", "aria-label": label },
      ...pictures.map((picture) => el("option", { value: picture.id, text: picture.label, selected: picture.id === chosen })));
    const under = choose("A picture", a), over = choose("B picture", b);
    const divide = el("input", { type: "range", min: "0", max: "100", value: "50", class: "ab-divide", "aria-label": "A/B divide" });
    const frame = el("div", { class: "ab-frame" });
    const cut = () => { const top = frame.querySelector(".ab-b"); if (top) top.style.clipPath = "inset(0 0 0 " + divide.value + "%)"; };
    const draw = () => {
      frame.replaceChildren(el("img", { src: contentURL(under.value), alt: "A", class: "ab-a" }), el("img", { src: contentURL(over.value), alt: "B", class: "ab-b" }));
      cut();
    };
    under.addEventListener("change", draw);
    over.addEventListener("change", draw);
    divide.addEventListener("input", cut);
    draw();
    return el("section", { class: "ab-compare", "aria-label": "A/B compare" },
      el("div", { class: "row" }, el("span", { class: "tag", text: "A" }), under, el("span", { class: "tag", text: "B" }), over), frame, divide);
  }

  // shown: a completed run's outputs, its pictures beside an A/B compare of its newest picture (B) against the picture
  // its request started from, else the picture drawn before it (A); a non-picture output stands as its link.
  function shown(host, completed, overgo) {
    const { el, contentURL, fmt } = overgo;
    const outputs = completed.outputs || [];
    if (completed.task !== "image-gen") {
      host.replaceChildren(...outputs.map((id) => overgo.artifactLink(id)));
      return;
    }
    const input = (request && request().input) || {};
    const started = [input.image, input.control && input.control.picture].filter(Boolean);
    const before = pictures.find((picture) => !outputs.includes(picture.id) && !started.includes(picture.id));
    for (const id of started) remember(id, "started from · " + fmt.shortID(id));
    for (const id of [...outputs].reverse()) remember(id, "run " + fmt.shortID(completed.run || id) + " · " + fmt.shortID(id));
    const a = started[0] || (before && before.id);
    host.replaceChildren(
      el("div", { class: "workbench-outputs" }, ...outputs.map((id) => el("img", { src: contentURL(id), alt: "Generated picture", class: "workbench-output" }))),
      ...(a && outputs.length ? [abCompare(overgo, a, outputs[0])] : []));
  }

  window.overgo.workflowWorkspace({ id: "generation", scope: "generation", preview: depthPreview, renderEvidence: shown });
})();
