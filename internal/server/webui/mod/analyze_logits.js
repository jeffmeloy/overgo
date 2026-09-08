/* Logit lens tab: runs a short greedy completion with n_probs and visualizes
   the model's per-position next-token distribution — top-k probabilities and
   full-vocabulary Shannon entropy (nats). Every value plotted is measured by
   the server; nothing is fitted or assumed about the distribution's shape. */
(function () {
  "use strict";
  let vocabSize = 0; // cached across runs to draw the entropy ceiling ln(vocab)

  window.overgo.registerTab({
    id: "lens",
    async mount(panel, overgo, seed) {
      const { el, clear, displayToken } = overgo;
      clear(panel);

      const maxTokens = el("input", { class: "keyfield w-90", type: "number", value: "24", min: "1", max: "128", "aria-label": "max tokens" });
      const topK = el("input", { class: "keyfield w-90", type: "number", value: "10", min: "1", max: "40", "aria-label": "top k" });
      const { prompt, out } = overgo.analysisSurface(panel, seed, {
        defaultPrompt: "The capital of France is", runLabel: "run lens", busy: "running…",
        fields: [["max tokens", maxTokens], ["top-k", topK]],
        execute: async (signal) => {
          await ensureVocabSize();
          render(await overgo.api.post("/completion", {
            prompt: prompt.value,
            n_predict: Number(maxTokens.value) || 24,
            n_probs: Number(topK.value) || 10,
            temperature: 0, // greedy: the lens inspects the argmax trajectory
            stream: false,
            cache_prompt: false,
          }, { signal }));
        },
      });

      async function ensureVocabSize() {
        if (vocabSize > 0) return;
        try { vocabSize = (await overgo.modelInfo()).model.vocabulary_size || 0; } catch (_) { /* optional */ }
      }

      function render(data) {
        const steps = data.completion_probabilities || [];
        if (steps.length === 0) {
          out.replaceChildren(el("div", { class: "note", text: "no per-token probabilities returned." }));
          return;
        }
        clear(out);

        const entropies = steps.map((s) => (s.entropy != null ? s.entropy : 0));
        const meanEntropy = entropies.reduce((a, b) => a + b, 0) / entropies.length;
        const reference = vocabSize > 1 ? Math.log(vocabSize) : null;

        // --- sequence entropy ------------------------------------------------
        out.appendChild(el("div", { class: "section-title", text: "Entropy across the sequence (nats)" }));
        const summary = el("div", { class: "lens-summary" },
          el("span", { class: "note", text: steps.length + " tokens" }),
          el("span", { class: "note", text: "mean entropy " + meanEntropy.toFixed(3) + " nats" }),
          reference != null ? el("span", { class: "note", text: "ceiling ln(vocab) = " + reference.toFixed(2) }) : null);
        out.appendChild(summary);
        out.appendChild(overgo.viz.sparkline(entropies, { height: 60, reference: reference, label: "H" }));

        // --- per-token distributions ----------------------------------------
        out.appendChild(el("div", { class: "section-title", text: "Per-token next-token distribution (top-k)" }));
        steps.forEach((step, index) => {
          const chosen = step.token;
          const top = step.top_logprobs || [];
          const entries = top.map((item) => ({
            label: displayToken(item.token),
            value: Math.exp(item.logprob),
            highlight: item.id === step.id,
            sub: (100 * Math.exp(item.logprob)).toFixed(1) + "%",
            title: JSON.stringify(item.token) + "  logprob " + item.logprob.toFixed(3),
          }));
          const card = el("div", { class: "lens-token" },
            el("div", { class: "head" },
              el("span", { class: "note", text: "#" + index }),
              el("span", { class: "chosen", text: displayToken(chosen) }),
              el("span", { class: "note", text: "H = " + (step.entropy != null ? step.entropy.toFixed(3) : "—") + " nats" }),
              step.logprob != null ? el("span", { class: "note", text: "p(chosen) = " + (100 * Math.exp(step.logprob)).toFixed(1) + "%" }) : null),
            overgo.viz.probBars(entries, { max: 1 }));
          out.appendChild(card);
        });
      }

    },
  });
})();
