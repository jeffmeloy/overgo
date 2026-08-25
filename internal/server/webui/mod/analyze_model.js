/* Model tab: renders GET /analyze/model — measured architecture facts and their
   exact derived ratios. No statistics, nothing assumed about the data. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "model",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      panel.appendChild(el("div", { class: "note", text: "loading /analyze/model…" }));

      let data;
      try {
        data = await overgo.modelInfo();
      } catch (err) {
        clear(panel);
        panel.appendChild(overgo.errorBanner(overgo.friendlyError(err)));
        return;
      }
      const m = data.model, d = data.derived;
      clear(panel);

      // --- facts -------------------------------------------------------------
      panel.appendChild(el("div", { class: "section-title", text: "Model" }));
      const facts = el("div", { class: "statgrid" });
      facts.append(
        overgo.stat("Architecture", m.architecture || "—"),
        overgo.stat("Parameters", fmt.compact(m.parameters), "(" + fmt.grouped(m.parameters) + ")"),
        overgo.stat("Size", fmt.bytes(m.size_bytes)),
        overgo.stat("File type", m.file_type || "—"),
        overgo.stat("Context length", fmt.grouped(m.context_length), "tokens"),
        overgo.stat("Vocabulary", fmt.grouped(m.vocabulary_size), m.vocabulary_type || ""),
        overgo.stat("Embedding dim", fmt.grouped(m.embedding_length)),
        overgo.stat("FFN dim", fmt.grouped(m.feed_forward_length)),
        overgo.stat("Blocks", m.block_count),
        overgo.stat("Attention heads", m.head_count),
        overgo.stat("KV heads", m.head_count_kv),
      );
      panel.appendChild(facts);

      // --- derived (exact ratios) -------------------------------------------
      panel.appendChild(el("div", { class: "section-title", text: "Derived (exact ratios)" }));
      const derived = el("div", { class: "statgrid" });
      if (d.head_dim != null) derived.appendChild(overgo.stat("Head dim", d.head_dim, "= embd / heads"));
      if (d.kv_group_size != null) {
        derived.appendChild(overgo.stat("KV group size", d.kv_group_size, d.grouped_query ? "GQA" : "MHA"));
      }
      if (d.params_per_block != null) derived.appendChild(overgo.stat("Params / block", fmt.compact(d.params_per_block)));
      if (d.avg_bits_per_weight != null) {
        derived.appendChild(overgo.stat("Avg bits / weight", d.avg_bits_per_weight.toFixed(2), "measured"));
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
