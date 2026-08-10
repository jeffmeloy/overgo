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

  // ramp: map t in [0,1] to a 3-stop sequential color (low = similar/near,
  // high = dissimilar/far). Fixed stops so it reads the same in both themes.
  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    const stops = [[13, 17, 27], [55, 211, 238], [240, 86, 122]];
    const seg = t < 0.5 ? 0 : 1;
    const local = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
    const a = stops[seg], b = stops[seg + 1];
    const c = a.map((v, i) => Math.round(v + (b[i] - v) * local));
    return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
  }

  // heatmap: an N×N matrix as a grid of colored cells (e.g. a distance matrix).
  // Values are shown as-is; the color encodes magnitude relative to `max`.
  function heatmap(matrix, opts) {
    opts = opts || {};
    const n = matrix.length;
    const max = opts.max != null ? opts.max : Math.max(1e-9, ...matrix.flat());
    const cell = Math.max(6, Math.min(22, Math.floor(420 / Math.max(1, n))));
    const size = n * cell;
    const node = svg("svg", { viewBox: "0 0 " + size + " " + size, width: Math.min(size, 460), height: Math.min(size, 460) });
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const value = matrix[i][j];
        const rect = svg("rect", { x: j * cell, y: i * cell, width: cell, height: cell, fill: ramp(value / max) });
        const title = document.createElementNS(NS, "title");
        title.textContent = "(" + i + "," + j + ") = " + value.toFixed(4);
        rect.appendChild(title);
        node.appendChild(rect);
      }
    }
    return node;
  }

  // graph: nodes at `coords` ([x,y] pairs) with `edges` ([i,j] pairs). Used for
  // the kNN neighbor graph laid out by non-metric MDS. Coordinates are used as
  // given (already a rank-based layout) — only rescaled to fit the viewport.
  function graph(coords, edges, opts) {
    opts = opts || {};
    const size = 380, pad = 22;
    const xs = coords.map((c) => c[0]), ys = coords.map((c) => c[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const sx = (v) => pad + (maxX > minX ? (v - minX) / (maxX - minX) : 0.5) * (size - 2 * pad);
    const sy = (v) => pad + (maxY > minY ? (v - minY) / (maxY - minY) : 0.5) * (size - 2 * pad);
    const node = svg("svg", { viewBox: "0 0 " + size + " " + size, width: "100%", height: size });
    for (const [a, b] of edges) {
      node.appendChild(svg("line", { x1: sx(coords[a][0]), y1: sy(coords[a][1]), x2: sx(coords[b][0]), y2: sy(coords[b][1]), stroke: "var(--line)", "stroke-width": 1 }));
    }
    coords.forEach((c, i) => {
      const dot = svg("circle", { cx: sx(c[0]), cy: sy(c[1]), r: 4, fill: opts.colors ? opts.colors[i] : "var(--acc)" });
      const title = document.createElementNS(NS, "title");
      title.textContent = opts.labels ? opts.labels[i] : String(i);
      dot.appendChild(title);
      node.appendChild(dot);
      if (opts.labels) {
        const label = svg("text", { x: sx(c[0]) + 6, y: sy(c[1]) + 3, "font-size": 10, fill: "var(--dim)" });
        label.textContent = opts.labels[i];
        node.appendChild(label);
      }
    });
    return node;
  }

  window.overgo = window.overgo || {};
  window.overgo.viz = { sparkline, probBars, heatmap, graph };
})();
