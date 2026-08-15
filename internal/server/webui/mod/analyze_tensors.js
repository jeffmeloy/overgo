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
    if (t.spectral_status === "deferred") return "def";
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
    label: "Tensors",
    section: "workbench",
    requires: "tensors",
    async mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);
      panel.appendChild(el("div", { class: "note", text: "loading /analyze/tensors…" }));

      let data;
      try {
        data = await overgo.api.get("/analyze/tensors");
      } catch (err) {
        clear(panel);
        panel.appendChild(overgo.errorBanner(overgo.friendlyError(err)));
        return;
      }
      const rows = data.tensors || [];
      clear(panel);

      panel.appendChild(el("div", { class: "section-title", text: "Tensor value statistics" }));
      panel.appendChild(el("div", {
        class: "note",
        text: data.count + " tensors · distribution-free (L-moments + energy) · sampled ≤ " +
          fmt.grouped(data.policy.max_samples_per_tensor) + " values/tensor · click a row for nearest-shape tensors",
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

      const detail = el("div", { style: "margin-top:14px" });
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
        const list = el("div", { style: "display:flex;flex-direction:column;gap:4px" });
        for (const n of (data.neighbors || [])) {
          const row = el("div", {
            style: "display:flex;gap:12px;align-items:center;font-size:12px;padding:4px 8px;border:1px solid var(--line-soft);border-radius:8px;cursor:pointer",
          });
          row.addEventListener("click", () => showNeighbors(n.name));
          row.append(
            el("span", { style: "flex:1", text: n.name }),
            el("span", { style: "opacity:.7", text: "d=" + n.distance.toFixed(4) }),
            el("span", { style: "opacity:.7", text: "τ₃=" + n.l_moments.Tau3.toFixed(3) }),
            el("span", { style: "opacity:.7", text: "H=" + n.values.normalized_energy_entropy.toFixed(3) }),
          );
          list.appendChild(row);
        }
        detail.appendChild(list);
      }

      function bar(fraction) {
        const clamped = Math.max(0, Math.min(1, fraction));
        const wrap = el("div", {
          style: "display:flex;align-items:center;gap:6px;justify-content:flex-end",
        });
        const track = el("div", {
          style: "width:64px;height:8px;border-radius:4px;background:var(--bg2);overflow:hidden",
        });
        track.appendChild(el("div", {
          style: "height:100%;width:" + (clamped * 100).toFixed(1) + "%;background:var(--acc)",
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
          const tr = el("tr", { style: "cursor:pointer" });
          tr.addEventListener("click", () => showNeighbors(t.name));
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
            effrankText(t),
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
