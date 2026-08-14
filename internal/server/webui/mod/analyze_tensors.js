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
    { key: "entropy", label: "Energy entropy", num: true, get: (t) => t.values.normalized_energy_entropy },
  ];

  function sci(value) {
    if (value === 0) return "0";
    const a = Math.abs(value);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return value.toExponential(2);
    return value.toFixed(4);
  }

  window.overgo.registerTab({
    id: "tensors",
    label: "Tensors",
    section: "workbench",
    requires: "tensors",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      panel.appendChild(el("div", { class: "note", text: "loading /analyze/tensors…" }));

      const data = await overgo.api.get("/analyze/tensors");
      const rows = data.tensors || [];
      clear(panel);

      panel.appendChild(el("div", { class: "section-title", text: "Tensor value statistics" }));
      panel.appendChild(el("div", {
        class: "note",
        text: data.count + " tensors · distribution-free (L-moments + energy) · sampled ≤ " +
          fmt.grouped(data.policy.max_samples_per_tensor) + " values/tensor",
      }));

      let sortKey = "name";
      let sortAsc = true;

      const table = el("table", { class: "mono", style: "width:100%;border-collapse:collapse;font-size:12px" });
      const head = el("tr", {});
      for (const col of COLUMNS) {
        const th = el("th", {
          text: col.label,
          style: "text-align:" + (col.num ? "right" : "left") +
            ";padding:6px 8px;border-bottom:1px solid var(--line-soft);cursor:pointer;white-space:nowrap",
        });
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

      function bar(fraction) {
        const clamped = Math.max(0, Math.min(1, fraction));
        const wrap = el("div", {
          style: "display:flex;align-items:center;gap:6px;justify-content:flex-end",
        });
        const track = el("div", {
          style: "width:64px;height:8px;border-radius:4px;background:var(--bg2);overflow:hidden",
        });
        track.appendChild(el("div", {
          style: "height:100%;width:" + (clamped * 100).toFixed(1) + "%;background:var(--accent)",
        }));
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
          const tr = el("tr", {});
          tr.appendChild(el("td", { text: t.name, style: "padding:5px 8px;border-bottom:1px solid var(--line-soft)" }));
          tr.appendChild(el("td", { text: t.storage, style: "padding:5px 8px;border-bottom:1px solid var(--line-soft)" }));
          const cells = [
            fmt.compact(t.elements),
            sci(t.median),
            sci(t.l_moments.L2),
            t.l_moments.Tau3.toFixed(3),
            t.l_moments.Tau4.toFixed(3),
            (t.values.zero_fraction * 100).toFixed(1),
            sci(t.values.max_absolute),
          ];
          for (const value of cells) {
            tr.appendChild(el("td", {
              text: value,
              style: "padding:5px 8px;text-align:right;border-bottom:1px solid var(--line-soft)",
            }));
          }
          const entropyCell = el("td", { style: "padding:5px 8px;border-bottom:1px solid var(--line-soft)" });
          entropyCell.appendChild(bar(t.values.normalized_energy_entropy));
          tr.appendChild(entropyCell);
          body.appendChild(tr);
        }
        for (const th of head.children) {
          const c = COLUMNS[Array.prototype.indexOf.call(head.children, th)];
          th.textContent = c.label + (c.key === sortKey ? (sortAsc ? " ▲" : " ▼") : "");
        }
      }
      draw();
    },
  });
})();
