/* Logit lens tab: runs a short greedy completion with n_probs and visualizes
   the model's per-position next-token distribution — top-k probabilities and
   full-vocabulary Shannon entropy (nats). Every value plotted is measured by
   the server; nothing is fitted or assumed about the distribution's shape. */
(function () {
  "use strict";
  let vocabSize = 0; // cached across runs to draw the entropy ceiling ln(vocab)

  window.overgo.registerTab({
    id: "lens",
    label: "Logit lens",
    section: "workbench",
    requires: "logits",
    async mount(panel, overgo) {
      const { el, clear } = overgo;
      clear(panel);

      const prompt = el("textarea", { class: "text", placeholder: "prompt to analyze…" });
      prompt.value = "The capital of France is";
      const maxTokens = el("input", { class: "keyfield", type: "number", value: "24", min: "1", max: "128", style: "width:90px" });
      const topK = el("input", { class: "keyfield", type: "number", value: "10", min: "1", max: "40", style: "width:90px" });
      const run = el("button", { class: "btn", onclick: execute }, "run lens");
      const controls = el("div", {},
        prompt,
        el("div", { class: "row", style: "margin:10px 0" },
          el("span", { class: "note", text: "max tokens" }), maxTokens,
          el("span", { class: "note", text: "top-k" }), topK,
          run));
      const out = el("div");
      panel.append(controls, out);

      async function ensureVocabSize() {
        if (vocabSize > 0) return;
        try { vocabSize = (await overgo.api.get("/analyze/model")).model.vocabulary_size || 0; } catch (_) { /* optional */ }
      }

      async function execute() {
        out.replaceChildren(el("div", { class: "note", text: "running…" }));
        run.disabled = true;
        try {
          await ensureVocabSize();
          const data = await overgo.api.post("/completion", {
            prompt: prompt.value,
            n_predict: Number(maxTokens.value) || 24,
            n_probs: Number(topK.value) || 10,
            temperature: 0, // greedy: the lens inspects the argmax trajectory
            stream: false,
            cache_prompt: false,
          });
          render(data);
        } catch (err) {
          const message = err.status === 401 ? "API key required — enter it in the top bar." : String(err.message || err);
          out.replaceChildren(overgo.errorBanner(message));
        } finally {
          run.disabled = false;
        }
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

      // displayToken: make whitespace-only / empty pieces visible without
      // altering the measured text elsewhere.
      function displayToken(text) {
        if (text === "" ) return "∅";
        if (/^\s+$/.test(text)) return "␠".repeat(text.length);
        return text.replace(/\n/g, "⏎");
      }
    },
  });
})();
