/* Attention tab: capture one layer's query/key over the prompt and show the
   exact per-head causal attention weights softmax(scale·Q·Kᵀ), recomputed on the
   host from tensors captured at the attention-op boundary. No CUDA kernel change.

   The weights are the model's own normalized attention, genuine probabilities in [0,1], so the heatmap
   uses a fixed 0..1 color scale. Row i (query token) shows how position i attends over keys 0..i. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "attention",
    async mount(panel, overgo, seed) {
      const { el, clear, displayToken } = overgo;
      clear(panel);

      const layer = el("input", { class: "keyfield w-80", type: "number", placeholder: "mid", min: "0" });
      const maxPos = el("input", { class: "keyfield w-80", type: "number", value: "32", min: "2", max: "48", "aria-label": "positions" });
      let current = null; // last response, kept so the head selector can redraw.
      const { prompt, out } = overgo.analysisSurface(panel, seed, {
        defaultPrompt: "The quick brown fox jumps over the lazy dog", runLabel: "capture", busy: "capturing…",
        fields: [["layer", layer], ["max tokens", maxPos]],
        note: "Exact softmax(scale·Q·Kᵀ) per head, recomputed on the host from captured query/key. Weights are causal (row i attends to keys 0..i) and sum to 1.",
        execute: async (signal) => {
          const request = { prompt: prompt.value, max_positions: Number(maxPos.value) || 32 };
          if (layer.value !== "") request.layer = Number(layer.value);
          current = await overgo.api.post("/analyze/attention", request, { signal });
          render();
        },
      });

      function render() {
        const data = current;
        clear(out);
        const labels = data.tokens.map((t) => (t.text ? displayToken(t.text) : String(t.id)));

        if (data.truncated) {
          out.appendChild(el("div", { class: "note amber",
            text: "Prompt truncated to " + data.positions + " of " + data.requested_positions + " tokens (max " + data.max_positions + ")." }));
        }

        const groupNote = data.group_size > 1 ? " · GQA group " + data.group_size : "";
        out.appendChild(el("div", { class: "lens-summary" },
          el("span", { class: "note", text: "layer " + data.layer + " / " + data.block_count }),
          el("span", { class: "note", text: data.positions + " tokens · head dim " + data.head_dim }),
          el("span", { class: "note", text: data.heads + " heads · " + data.kv_heads + " kv heads" + groupNote }),
          el("span", { class: "note", text: "scale " + data.scale.toFixed(4) })));

        const head = el("select", { class: "keyfield w-110", "aria-label": "head", onchange: draw });
        for (let h = 0; h < data.heads; h++) head.appendChild(el("option", { value: String(h) }, "head " + h));
        out.appendChild(el("div", { class: "row my-8" },
          el("span", { class: "note", text: "head" }), head));

        const map = el("div");
        out.appendChild(map);
        // Token legend so the axes of the (unlabeled) heatmap are readable.
        out.appendChild(el("div", { class: "section-title", text: "Token index → piece" }));
        out.appendChild(el("div", { class: "note mono-lines",
          text: labels.map((t, i) => i + ":" + t).join("  ") }));
        out.appendChild(el("div", { class: "note mt-8",
          text: "Rows = query token (i), columns = key token (j). Cell (i,j) is the attention weight from i to j; the upper triangle is zero by causality. Hover a cell for the exact value." }));

        function draw() {
          const h = Number(head.value);
          map.replaceChildren(
            el("div", { class: "section-title", text: "Attention weights — head " + h + " (query × key)" }),
            // Fixed 0..1 scale: these are probabilities, not an inferred range.
            overgo.viz.heatmap(data.weights[h], { max: 1, labels: labels }));
        }
        draw();
      }

    },
  });
})();
