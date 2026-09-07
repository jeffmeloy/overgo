/* Hidden states tab: capture the residual-stream vectors at one layer over the
   prompt tokens and show their distribution-free structure — a distance matrix
   (heatmap) and a kNN neighbor graph laid out by non-metric MDS.

   The dissimilarity is an explicit choice, defaulting to Spearman rank correlation (no scale or
   Euclidean assumption); the matrix and neighbor graph are primary, the 2D scatter a rank-preserving
   summary whose faithfulness is reported as MDS stress. No PCA, no t-SNE. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "states",
    async mount(panel, overgo, seed) {
      const { el, clear, displayToken } = overgo;
      clear(panel);

      const metric = el("select", { class: "keyfield w-150", "aria-label": "metric" },
        el("option", { value: "spearman" }, "spearman (rank)"),
        el("option", { value: "cosine" }, "cosine"),
        el("option", { value: "euclidean" }, "euclidean"));
      const layer = el("input", { class: "keyfield w-80", type: "number", placeholder: "mid", min: "0" });
      const k = el("input", { class: "keyfield w-80", type: "number", placeholder: "auto", min: "1", max: "20" });
      const maxPos = el("input", { class: "keyfield w-80", type: "number", value: "48", min: "2", max: "64", "aria-label": "positions" });
      const { prompt, out } = overgo.analysisSurface(panel, seed, {
        defaultPrompt: "The quick brown fox jumps over the lazy dog", runLabel: "capture", busy: "capturing…",
        fields: [["metric", metric], ["layer", layer], ["k", k], ["max tokens", maxPos]],
        note: "Distribution-free: the metric is explicit (default = rank correlation); the layout uses only the rank order of distances (no PCA/t-SNE).",
        execute: async (signal) => {
          const request = { prompt: prompt.value, metric: metric.value, max_positions: Number(maxPos.value) || 48 };
          if (layer.value !== "") request.layer = Number(layer.value);
          if (k.value !== "") request.k = Number(k.value);
          render(await overgo.api.post("/analyze/states", request, { signal }));
        },
      });

      function render(data) {
        clear(out);
        const labels = data.tokens.map((t) => (t.text ? displayToken(t.text) : String(t.id)));
        // Color nodes by sequence position (early → late) so the graph shows how
        // token order maps onto the structure.
        const colors = data.tokens.map((_, i) => {
          const t = data.positions > 1 ? i / (data.positions - 1) : 0;
          return "hsl(" + Math.round(210 - 150 * t) + ", 65%, 55%)";
        });

        if (data.truncated) {
          out.appendChild(el("div", { class: "note amber",
            text: "Prompt truncated to " + data.positions + " of " + data.requested_positions + " tokens (max " + data.max_positions + ")." }));
        }

        out.appendChild(el("div", { class: "lens-summary" },
          el("span", { class: "note", text: "layer " + data.layer + " / " + data.block_count }),
          el("span", { class: "note", text: data.positions + " tokens · dim " + data.width }),
          el("span", { class: "note", text: "metric: " + data.metric }),
          el("span", { class: "note", text: "k = " + data.k }),
          el("span", { class: "note", text: "MDS stress " + data.stress.toFixed(3) + " (" + data.layout_iterations + " iters)" })));

        out.appendChild(el("div", { class: "section-title", text: "Distance matrix — " + data.metric + " (token × token)" }));
        out.appendChild(overgo.viz.heatmap(data.distance, { labels: labels })); // auto-scaled; no assumed range

        out.appendChild(el("div", { class: "section-title", text: "Neighbor graph (kNN edges, non-metric-MDS layout)" }));
        const edges = [];
        data.neighbors.forEach((neighbors, i) => neighbors.forEach((j) => edges.push([i, j])));
        out.appendChild(overgo.viz.graph(data.layout, edges, { labels: labels, colors: colors }));
        out.appendChild(el("div", { class: "note mt-8",
          text: "The matrix and neighbor graph are primary. The 2D layout is a rank-preserving summary only — read it together with the stress above (higher stress = the 2D placement is a poorer summary)." }));
      }

    },
  });
})();
