/* Model tab: renders GET /analyze/model — measured architecture facts and their
   exact derived ratios. No statistics, nothing assumed about the data. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "model",
    label: "Model",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      panel.appendChild(el("div", { class: "note", text: "loading /analyze/model…" }));

      const data = await overgo.api.get("/analyze/model");
      const m = data.model, d = data.derived;
      clear(panel);

      function stat(k, v, sub) {
        return el("div", { class: "stat" },
          el("div", { class: "k", text: k }),
          el("div", { class: "v" }, String(v), sub ? el("small", { text: " " + sub }) : null));
      }

      // --- facts -------------------------------------------------------------
      panel.appendChild(el("div", { class: "section-title", text: "Model" }));
      const facts = el("div", { class: "statgrid" });
      facts.append(
        stat("Architecture", m.architecture || "—"),
        stat("Parameters", fmt.compact(m.parameters), "(" + fmt.grouped(m.parameters) + ")"),
        stat("Size", fmt.bytes(m.size_bytes)),
        stat("File type", m.file_type || "—"),
        stat("Context length", fmt.grouped(m.context_length), "tokens"),
        stat("Vocabulary", fmt.grouped(m.vocabulary_size), m.vocabulary_type || ""),
        stat("Embedding dim", fmt.grouped(m.embedding_length)),
        stat("FFN dim", fmt.grouped(m.feed_forward_length)),
        stat("Blocks", m.block_count),
        stat("Attention heads", m.head_count),
        stat("KV heads", m.head_count_kv),
      );
      panel.appendChild(facts);

      // --- derived (exact ratios) -------------------------------------------
      panel.appendChild(el("div", { class: "section-title", text: "Derived (exact ratios)" }));
      const derived = el("div", { class: "statgrid" });
      if (d.head_dim != null) derived.appendChild(stat("Head dim", d.head_dim, "= embd / heads"));
      if (d.kv_group_size != null) {
        derived.appendChild(stat("KV group size", d.kv_group_size, d.grouped_query ? "GQA" : "MHA"));
      }
      if (d.params_per_block != null) derived.appendChild(stat("Params / block", fmt.compact(d.params_per_block)));
      if (d.avg_bits_per_weight != null) {
        derived.appendChild(stat("Avg bits / weight", d.avg_bits_per_weight.toFixed(2), "measured"));
      }
      panel.appendChild(derived);

      // --- capabilities ------------------------------------------------------
      panel.appendChild(el("div", { class: "section-title", text: "Capabilities" }));
      const caps = el("div", { class: "row" });
      for (const capability of (data.capabilities || [])) caps.appendChild(el("span", { class: "tag", text: capability }));
      panel.appendChild(caps);

      // --- chat template -----------------------------------------------------
      if (m.chat_template) {
        panel.appendChild(el("div", { class: "section-title", text: "Chat template (" + d.chat_template_bytes + " bytes)" }));
        panel.appendChild(el("pre", {
          class: "mono",
          style: "background:var(--bg2);border:1px solid var(--line-soft);border-radius:10px;padding:12px;overflow:auto;max-height:320px;font-size:12px",
          text: m.chat_template,
        }));
      }
    },
  });
})();
