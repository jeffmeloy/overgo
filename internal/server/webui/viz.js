/* overgo_gui viz: dependency-free SVG/DOM primitives. Draws only what was
   measured — no smoothing, no fitted curves, no interpolation beyond straight
   segments between the actual sample points. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    for (const name in attrs) node.setAttribute(name, attrs[name]);
    return node;
  }

  // sparkline: one straight segment per adjacent measured pair, scaled to
  // [0, max]. An optional reference value (e.g. ln(vocab) for entropy) is drawn
  // as a dashed line so the reader sees the ceiling without any curve fitting.
  function sparkline(values, opts) {
    opts = opts || {};
    const height = opts.height || 48;
    const width = opts.width || Math.max(140, values.length * 12);
    const pad = 5;
    const max = opts.max != null ? opts.max : Math.max(1e-9, ...values, opts.reference || 0);
    const node = svg("svg", { viewBox: "0 0 " + width + " " + height, width: "100%", height: height, preserveAspectRatio: "none" });
    const y = (v) => height - pad - (Math.max(0, v) / max) * (height - 2 * pad);
    if (opts.reference != null) {
      node.appendChild(svg("line", { x1: 0, y1: y(opts.reference), x2: width, y2: y(opts.reference), stroke: "var(--line)", "stroke-dasharray": "3 3", "stroke-width": 1 }));
    }
    const n = values.length;
    if (n > 0) {
      const step = n > 1 ? (width - 2 * pad) / (n - 1) : 0;
      let d = "";
      values.forEach((v, i) => { d += (i ? "L" : "M") + (pad + i * step).toFixed(1) + " " + y(v).toFixed(1) + " "; });
      node.appendChild(svg("path", { d: d.trim(), fill: "none", stroke: "var(--acc)", "stroke-width": 1.5 }));
      values.forEach((v, i) => {
        const dot = svg("circle", { cx: pad + i * step, cy: y(v), r: 2, fill: "var(--acc)" });
        const title = document.createElementNS(NS, "title");
        title.textContent = (opts.label ? opts.label + " " : "") + "[" + i + "] = " + v.toFixed(4);
        dot.appendChild(title);
        node.appendChild(dot);
      });
    }
    return node;
  }

  // probBars: horizontal bars for measured probabilities. Each entry is
  // {label, value in [0,1], sub?, highlight?, title?}.
  function probBars(entries, opts) {
    opts = opts || {};
    const wrap = document.createElement("div");
    const max = opts.max != null ? opts.max : Math.max(1e-9, ...entries.map((e) => e.value));
    for (const entry of entries) {
      const row = document.createElement("div");
      row.className = "pbar-row";
      if (entry.title) row.title = entry.title;

      const label = document.createElement("span");
      label.className = "pbar-label mono";
      label.textContent = entry.label;

      const track = document.createElement("span");
      track.className = "pbar-track";
      const fill = document.createElement("span");
      fill.className = "pbar-fill" + (entry.highlight ? " hi" : "");
      fill.style.width = (100 * Math.min(1, entry.value / max)).toFixed(1) + "%";
      track.appendChild(fill);

      const value = document.createElement("span");
      value.className = "pbar-val mono";
      value.textContent = entry.sub != null ? entry.sub : (100 * entry.value).toFixed(1) + "%";

      row.append(label, track, value);
      wrap.appendChild(row);
    }
    return wrap;
  }

  window.overgo = window.overgo || {};
  window.overgo.viz = { sparkline, probBars };
})();
