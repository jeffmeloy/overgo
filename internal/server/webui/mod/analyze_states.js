/* Hidden states tab: capture the residual-stream vectors at one layer over the
   prompt tokens and show their distribution-free structure — a distance matrix
   (heatmap) and a kNN neighbor graph laid out by non-metric MDS.

   The dissimilarity is an explicit choice, defaulting to Spearman rank
   correlation (invariant to any monotone per-coordinate transform — no scale or
   Euclidean assumption). The matrix and neighbor graph are the primary,
   least-assumptive views; the 2D scatter is only a rank-preserving summary whose
   faithfulness is reported as MDS stress. No PCA, no t-SNE. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "states",
    label: "Hidden states",
    async mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);

      const prompt = el("textarea", { class: "text", placeholder: "prompt to analyze…" });
      prompt.value = "The quick brown fox jumps over the lazy dog";
      const metric = el("select", { class: "keyfield", style: "width:150px" },
        el("option", { value: "spearman" }, "spearman (rank)"),
        el("option", { value: "cosine" }, "cosine"),
        el("option", { value: "euclidean" }, "euclidean"));
      const layer = el("input", { class: "keyfield", type: "number", placeholder: "mid", min: "0", style: "width:80px" });
      const k = el("input", { class: "keyfield", type: "number", placeholder: "auto", min: "1", max: "20", style: "width:80px" });
      const maxPos = el("input", { class: "keyfield", type: "number", value: "48", min: "2", max: "64", style: "width:80px" });
      const run = el("button", { class: "btn", onclick: execute }, "capture");
      panel.append(
        prompt,
        el("div", { class: "row", style: "margin:10px 0" },
          el("span", { class: "note", text: "metric" }), metric,
          el("span", { class: "note", text: "layer" }), layer,
          el("span", { class: "note", text: "k" }), k,
          el("span", { class: "note", text: "max tokens" }), maxPos,
          run),
        el("div", { class: "note", text: "Distribution-free: the metric is explicit (default = rank correlation); the layout uses only the rank order of distances (no PCA/t-SNE)." }));
      const out = el("div");
      panel.appendChild(out);

      async function execute() {
        out.replaceChildren(el("div", { class: "note", text: "capturing…" }));
        run.disabled = true;
        try {
          const request = { prompt: prompt.value, metric: metric.value, max_positions: Number(maxPos.value) || 48 };
          if (layer.value !== "") request.layer = Number(layer.value);
          if (k.value !== "") request.k = Number(k.value);
          render(await overgo.api.post("/analyze/states", request));
        } catch (err) {
          const message = err.status === 401 ? "API key required — enter it in the top bar." : String(err.message || err);
          out.replaceChildren(overgo.errorBanner(message));
        } finally {
          run.disabled = false;
        }
      }

      function render(data) {
        clear(out);
        const labels = data.tokens.map((t) => displayToken(t.text) || String(t.id));

        out.appendChild(el("div", { class: "lens-summary" },
          el("span", { class: "note", text: "layer " + data.layer + " / " + data.block_count }),
          el("span", { class: "note", text: data.positions + " tokens · dim " + data.width }),
          el("span", { class: "note", text: "metric: " + data.metric }),
          el("span", { class: "note", text: "k = " + data.k }),
          el("span", { class: "note", text: "MDS stress " + data.stress.toFixed(3) + " (" + data.layout_iterations + " iters)" })));

        out.appendChild(el("div", { class: "section-title", text: "Distance matrix — " + data.metric + " (token × token)" }));
        out.appendChild(overgo.viz.heatmap(data.distance)); // auto-scaled; no assumed range

        out.appendChild(el("div", { class: "section-title", text: "Neighbor graph (kNN edges, non-metric-MDS layout)" }));
        const edges = [];
        data.neighbors.forEach((neighbors, i) => neighbors.forEach((j) => edges.push([i, j])));
        out.appendChild(overgo.viz.graph(data.layout, edges, { labels: labels }));
        out.appendChild(el("div", { class: "note", style: "margin-top:8px",
          text: "The matrix and neighbor graph are primary. The 2D layout is a rank-preserving summary only — read it together with the stress above (higher stress = the 2D placement is a poorer summary)." }));
      }

      function displayToken(text) {
        if (!text) return "";
        if (/^\s+$/.test(text)) return "␠";
        return text.replace(/\n/g, "⏎");
      }
    },
  });
})();
