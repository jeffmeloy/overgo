/* Hidden states tab: capture the residual-stream vectors at one layer over the
   prompt tokens and show their distribution-free structure — a cosine distance
   matrix (heatmap) and a kNN neighbor graph laid out by non-metric MDS. No PCA,
   no t-SNE: the layout depends only on the rank order of distances. */
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
      const layer = el("input", { class: "keyfield", type: "number", placeholder: "mid", min: "0", style: "width:90px" });
      const k = el("input", { class: "keyfield", type: "number", value: "5", min: "1", max: "20", style: "width:80px" });
      const maxPos = el("input", { class: "keyfield", type: "number", value: "48", min: "2", max: "64", style: "width:80px" });
      const run = el("button", { class: "btn", onclick: execute }, "capture");
      panel.append(
        prompt,
        el("div", { class: "row", style: "margin:10px 0" },
          el("span", { class: "note", text: "layer" }), layer,
          el("span", { class: "note", text: "k (neighbors)" }), k,
          el("span", { class: "note", text: "max tokens" }), maxPos,
          run),
        el("div", { class: "note", text: "Distribution-free: exact distances + rank-only non-metric-MDS layout (no PCA/t-SNE)." }));
      const out = el("div");
      panel.appendChild(out);

      async function execute() {
        out.replaceChildren(el("div", { class: "note", text: "capturing…" }));
        run.disabled = true;
        try {
          const request = { prompt: prompt.value, k: Number(k.value) || 5, max_positions: Number(maxPos.value) || 48 };
          if (layer.value !== "") request.layer = Number(layer.value);
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
          el("span", { class: "note", text: "MDS stress " + data.stress.toFixed(3) }),
          el("span", { class: "note", text: "method: " + data.method })));

        out.appendChild(el("div", { class: "section-title", text: "Cosine distance (token × token)" }));
        out.appendChild(overgo.viz.heatmap(data.cosine, { max: 2 }));

        out.appendChild(el("div", { class: "section-title", text: "Neighbor graph (non-metric-MDS layout, kNN edges)" }));
        const edges = [];
        data.neighbors.forEach((neighbors, i) => neighbors.forEach((j) => edges.push([i, j])));
        out.appendChild(overgo.viz.graph(data.layout, edges, { labels: labels }));
      }

      function displayToken(text) {
        if (!text) return "";
        if (/^\s+$/.test(text)) return "␠";
        return text.replace(/\n/g, "⏎");
      }
    },
  });
})();
