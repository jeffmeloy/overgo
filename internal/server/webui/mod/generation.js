(function () {
  "use strict";
  // The generation tab is MOSS's workbench as it is every media model's page: a model's controls and presets, a depth
  // section's preview before its run, and a run's pictures shown and compared.

  // depthPreview: a depth section's map as its run makes it, as the MOSS Ultra workflow's PreviewImage shows it, before
  // any run; offered while the page shows a depth control whose picture is chosen, the one the map is made from
  // (moss-page/faults). The estimate runs once per press, never again on its own. A map is shown only while the form
  // still asks what it was made from, the control's picture and the request's size: a change to them clears it, and an
  // answer that comes after one is not shown (SQA-493: picture A's map had stayed beside a picture B the run reads).
  function depthPreview(host, composed, overgo) {
    const { el, api } = overgo;
    const shown = el("div", { class: "depth-preview" });
    const open = el("button", { class: "btn alt", type: "button", text: "Preview depth map" });
    const madeFrom = () => {
      const asked = composed().input || {};
      return JSON.stringify([(asked.control || {}).picture || "", asked.width || 0, asked.height || 0, asked.canvas || null]);
    };
    let showing = null;
    open.addEventListener("click", async () => {
      open.disabled = true;
      const asked = madeFrom();
      showing = asked;
      shown.replaceChildren(el("div", { class: "note", role: "status", text: "Estimating depth…" }));
      try {
        const preview = await api.post("/generation/preview", { ...composed(), position: 0, limit: 1 });
        if (madeFrom() === asked) shown.replaceChildren(el("img", { src: overgo.contentURL(preview.output), alt: "Depth map", class: "depth-preview-image" }));
      } catch (err) {
        if (madeFrom() === asked) shown.replaceChildren(overgo.failure(err));
      } finally { open.disabled = false; }
    });
    host.replaceChildren(el("div", { class: "row" }, open), shown);
    const panel = host.parentElement;
    const picked = (control) => { try { return !!(JSON.parse(control.value || "null") || {}).picture; } catch (_) { return false; } };
    // Set only when it changes: a set queues a record for the observer below even where the value stays, which would
    // call offer again without end.
    const offer = () => {
      const control = panel.querySelector('fieldset[data-control="control"]');
      const hide = !control || !!control.closest("[hidden]") || !picked(control);
      if (host.hidden !== hide) host.hidden = hide;
      if (showing !== null && madeFrom() !== showing) {
        showing = null;
        shown.replaceChildren();
      }
    };
    new MutationObserver(offer).observe(panel, { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden"] });
    for (const name of ["input", "change"]) panel.addEventListener(name, offer);
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

  // promptsUsed: what a run's picture was drawn from, as the MOSS Ultra workflow shows it beside each section's picture
  // ("Prompt used", "Negative prompt used"), each offered back as the manual prompt: the prompt mode text, which shows
  // and sends the prompt as written and no layout (choiceShows), so the next run draws the text as written
  // (moss-prompts-used, moss-prompt-mode).
  function promptsUsed(host, used, overgo) {
    const { el } = overgo;
    const field = (name) => [...(host.closest(".panel") || document).querySelectorAll("label.control")]
      .find((label) => label.querySelector("span")?.textContent === name)?.querySelector("input,select,textarea");
    const set = (input, value) => { if (input) { input.value = value; input.dispatchEvent(new Event("input", { bubbles: true })); } };
    const manual = () => set(field("prompt mode"), "text");
    const shownText = (title, text, label, name) => {
      const offer = el("button", { class: "btn alt", type: "button", text: label });
      offer.addEventListener("click", () => { manual(); set(field(name), text); });
      return el("div", { class: "prompt-used" }, el("div", { class: "section-title", text: title }), el("pre", { class: "mono", text: text || "(none)" }), offer);
    };
    return el("section", { class: "prompts-used", "aria-label": "prompts used" },
      shownText("Prompt used", used.prompt, "Use as the prompt", "prompt"),
      shownText("Negative prompt used", used.negative_prompt, "Use as the negative prompt", "negative prompt"));
  }

  // promptEnhancementSchema is the stored schema of a prompt enhancement record, the server's (prompt_enhance.go).
  const promptEnhancementSchema = "overgo/prompt-enhancement/v1";

  // recordedInput: the request a completed run's record keeps, its first JSON input that is no prompt enhancement; none
  // where the run has no record.
  // A request kept as its digest (one above a mebibyte) names no picture, so no source is offered for it.
  async function recordedInput(completed, overgo) {
    if (!completed.run) return {};
    const run = await overgo.api.get("/runs?id=" + encodeURIComponent(completed.run));
    // A prompt enhancement the page accepted is a source the run cites beside its request, told apart by its stored
    // schema (prompt_enhance.go promptEnhancementSchema), whichever sorts first (SQA-470).
    const schemas = run.input_schemas || {};
    for (const id of run.inputs || []) {
      if (schemas[id] === promptEnhancementSchema) continue;
      const blob = await overgo.api.blob(overgo.contentURL(id));
      if (blob.type.includes("json")) return JSON.parse(await blob.text());
    }
    return {};
  }

  // shown: a completed run's outputs, its pictures beside an A/B compare: a sharpened run's picture as drawn (A) and
  // sharpened (B), as each MOSS Ultra section's comparer shows them (moss-sharpened-pair), else its picture (B) against
  // the picture its request started from or the picture drawn before it (A); and what its picture was drawn from where
  // the run says (promptsUsed); a non-picture output stands as its link. The pictures a request started from are the
  // ones sent, not the form's as the run ends, and for a run this page did not send (followed again after a reload)
  // the ones its recorded request names (recordedInput, SQA-463).
  async function shown(host, completed, overgo, sent) {
    const { el, contentURL, fmt } = overgo;
    let outputs = completed.outputs || [];
    if (completed.task !== "image-gen") {
      host.replaceChildren(...outputs.map((id) => overgo.artifactLink(id)));
      return;
    }
    // A run's outputs are its pictures and, where it says what they were drawn from, that document.
    const fetched = await Promise.all(outputs.map(async (id) => ({ id, blob: await overgo.api.blob(contentURL(id)) })));
    const drawnFrom = fetched.find((output) => output.blob.type.includes("json"));
    const used = drawnFrom ? JSON.parse(await drawnFrom.blob.text()) : null;
    outputs = fetched.filter((output) => output !== drawnFrom).map((output) => output.id);
    const input = sent || await recordedInput(completed, overgo);
    const started = [input.image, input.control && input.control.picture].filter(Boolean);
    const before = pictures.find((picture) => !outputs.includes(picture.id) && !started.includes(picture.id));
    for (const id of started) remember(id, "started from · " + fmt.shortID(id));
    // A sharpened run's outputs are its picture as drawn, then sharpened (moss-sharpened-pair).
    const paired = outputs.length === 2;
    const roles = paired ? ["drawn · ", "sharpened · "] : ["", ""];
    for (const [index, id] of [...outputs.entries()].reverse()) remember(id, roles[index] + "run " + fmt.shortID(completed.run || id) + " · " + fmt.shortID(id));
    const a = paired ? outputs[0] : started[0] || (before && before.id);
    host.replaceChildren(
      el("div", { class: "workbench-outputs" }, ...outputs.map((id) => el("img", { src: contentURL(id), alt: "Generated picture", class: "workbench-output" }))),
      ...(used ? [promptsUsed(host, used, overgo)] : []),
      ...(a && outputs.length ? [abCompare(overgo, a, outputs[outputs.length - 1])] : []));
  }

  window.overgo.workflowWorkspace({ id: "generation", scope: "generation", preview: depthPreview, renderEvidence: shown });
})();
