/* Tensors tab: renders GET /analyze/tensors — a distribution-free value profile
   for every tensor in the loaded model. Columns are measured functionals of the
   empirical distribution (robust L-moments and energy descriptors), never a
   fitted shape: L-scale is a dispersion that exists under heavy tails, tau3/tau4
   are bounded L-skewness/kurtosis, and energy entropy reads how concentrated the
   squared magnitudes are (0 = one element carries all energy, 1 = uniform).
   Click a column header to sort; sampling is bounded server-side. */
(function () {
  "use strict";

  const COLUMNS = [
    { key: "name", label: "Tensor", num: false, get: (t) => t.name },
    { key: "storage", label: "Type", num: false, get: (t) => t.storage },
    { key: "elements", label: "Elements", num: true, get: (t) => t.elements },
    { key: "median", label: "Median", num: true, get: (t) => t.median },
    { key: "l_scale", label: "L-scale", num: true, get: (t) => t.l_moments.L2 },
    { key: "tau3", label: "L-skew τ₃", num: true, get: (t) => t.l_moments.Tau3 },
    { key: "tau4", label: "L-kurt τ₄", num: true, get: (t) => t.l_moments.Tau4 },
    { key: "zero", label: "Zero %", num: true, get: (t) => t.values.zero_fraction },
    { key: "maxabs", label: "Max |·|", num: true, get: (t) => t.values.max_absolute },
    { key: "effrank", label: "Eff. rank", num: true, get: (t) => (t.spectral_status === "computed" ? t.effective_rank : -1) },
    { key: "entropy", label: "Energy entropy", num: true, get: (t) => t.values.normalized_energy_entropy },
  ];

  function effrankText(t) {
    if (t.spectral_status === "computed") return t.effective_rank.toFixed(3);
    if (t.spectral_status === "deferred") return "deferred"; // the read budget could not cover its spectrum
    return "—"; // not-applicable (1-D / non-matrix) or spectral off
  }

  function sci(value) {
    if (value === 0) return "0";
    const a = Math.abs(value);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return value.toExponential(2);
    return value.toFixed(4);
  }

  window.overgo.registerTab({
    id: "tensors",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      const data = await overgo.load((signal) => overgo.api.get("/analyze/tensors", { signal }), { loading: "Reading tensor statistics…" });
      if (!data) return;
      let rows = data.tensors || [];

      panel.appendChild(el("div", { class: "section-title", text: "Tensor value statistics" }));
      panel.appendChild(el("div", { class: "note", text: data.count + " tensors · distribution-free (L-moments + energy) · sampled ≤ " +
        fmt.grouped(data.policy.max_samples_per_tensor) + " values/tensor · click a row for nearest-shape tensors" }));
      // The first view is the sampled pass; effective ranks follow when the full pass ends.
      const spectra = el("div", { class: "note", role: "status" });
      panel.appendChild(spectra);
      function showSpectra(answer) {
        spectra.hidden = answer.spectra === "complete";
        spectra.textContent = answer.spectra === "failed" ? "Effective ranks are unavailable: " + answer.spectra_failure
          : "Computing effective ranks; the table updates when they are ready.";
      }
      showSpectra(data);

      let sortKey = "name";
      let sortAsc = true;

      const table = el("table", { class: "mono tensor-table" });
      const head = el("tr", {});
      for (const col of COLUMNS) {
        const th = el("th", { text: col.label, class: "cell head" + (col.num ? " num" : "") });
        th.addEventListener("click", () => {
          if (sortKey === col.key) sortAsc = !sortAsc;
          else { sortKey = col.key; sortAsc = col.num ? false : true; }
          draw();
        });
        head.appendChild(th);
      }
      table.appendChild(head);
      const body = el("tbody", {});
      table.appendChild(body);
      panel.appendChild(table);

      const detail = el("div", { class: "mt-14" });
      panel.appendChild(detail);

      async function showNeighbors(name) {
        clear(detail);
        detail.appendChild(el("div", { class: "note", text: "finding tensors similar to " + name + "…" }));
        let data;
        try {
          data = await overgo.api.get("/analyze/tensors/similar?name=" + encodeURIComponent(name) + "&k=8");
        } catch (e) {
          clear(detail);
          detail.appendChild(el("div", { class: "note", text: "similar lookup failed: " + e.message }));
          return;
        }
        clear(detail);
        detail.appendChild(el("div", { class: "section-title", text: "Nearest to " + name + " — by distribution shape (" + (data.metric || "rank") + ", d in [0,1])" }));
        const list = el("div", { class: "stack" });
        for (const n of (data.neighbors || [])) {
          const row = el("div", {
            class: "neighbour-row",
          });
          row.addEventListener("click", () => showNeighbors(n.name));
          row.append(
            el("span", { class: "grow", text: n.name }),
            el("span", { class: "faint", text: "d=" + n.distance.toFixed(4) }),
            el("span", { class: "faint", text: "τ₃=" + n.l_moments.Tau3.toFixed(3) }),
            el("span", { class: "faint", text: "H=" + n.values.normalized_energy_entropy.toFixed(3) }),
          );
          list.appendChild(row);
        }
        detail.appendChild(list);
      }

      function bar(fraction) {
        const clamped = Math.max(0, Math.min(1, fraction));
        const wrap = el("div", { class: "meter" });
        const track = el("div", { class: "meter-track" });
        const fill = el("div", { class: "meter-fill" });
        fill.style.width = (clamped * 100).toFixed(1) + "%"; // the one measured size, set as a property
        track.appendChild(fill);
        wrap.append(el("span", { text: clamped.toFixed(3) }), track);
        return wrap;
      }

      function draw() {
        const col = COLUMNS.find((c) => c.key === sortKey);
        const sorted = rows.slice().sort((a, b) => {
          const av = col.get(a), bv = col.get(b);
          const cmp = col.num ? av - bv : String(av).localeCompare(String(bv));
          return sortAsc ? cmp : -cmp;
        });
        clear(body);
        for (const t of sorted) {
          const tr = el("tr", { class: "clickable" });
          tr.addEventListener("click", () => showNeighbors(t.name));
          tr.appendChild(el("td", { text: t.name, class: "cell" }));
          tr.appendChild(el("td", { text: t.storage, class: "cell" }));
          const cells = [
            fmt.compact(t.elements),
            sci(t.median),
            sci(t.l_moments.L2),
            t.l_moments.Tau3.toFixed(3),
            t.l_moments.Tau4.toFixed(3),
            (t.values.zero_fraction * 100).toFixed(1),
            sci(t.values.max_absolute),
            effrankText(t),
          ];
          for (const value of cells) {
            tr.appendChild(el("td", { text: value, class: "cell num", }));
          }
          const entropyCell = el("td", { class: "cell" });
          entropyCell.appendChild(bar(t.values.normalized_energy_entropy));
          tr.appendChild(entropyCell);
          body.appendChild(tr);
        }
        for (const th of head.children) {
          const c = COLUMNS[Array.prototype.indexOf.call(head.children, th)];
          // The sorted column says so to assistive technology and draws its direction.
          th.textContent = c.label;
          if (c.key === sortKey) th.setAttribute("aria-sort", sortAsc ? "ascending" : "descending");
          else th.removeAttribute("aria-sort");
        }
      }
      draw();
      if (data.spectra === "pending") {
        overgo.api.get("/analyze/tensors?wait=spectra", { signal: overgo.signal }).then((answer) => {
          rows = answer.tensors || [];
          showSpectra(answer);
          draw();
        }, (err) => { if (err.name !== "AbortError") spectra.replaceChildren(overgo.failure(err)); });
      }
    },
  });
})();
