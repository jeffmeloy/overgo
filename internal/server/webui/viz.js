/* overgo_gui viz: dependency-free SVG/DOM primitives. Draws only what was
   measured — no smoothing, no fitted curves, no interpolation beyond straight
   segments between the actual sample points. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  let legendSeq = 0; // unique gradient ids so multiple legends never collide
  function svg(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    for (const name in attrs) node.setAttribute(name, attrs[name]);
    return node;
  }

  // sparkline: one straight segment per adjacent measured pair, scaled to [0, max]; an optional
  // reference value (ln(vocab) for entropy) is drawn as a dashed ceiling, no curve fitting.
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

  // signedSeries: raw measured points with a visible zero axis.
  function signedSeries(series, opts) {
    opts = opts || {};
    const height = opts.height || 120;
    const width = opts.width || 520;
    const pad = 10;
    const values = series.flatMap((item) => item.values);
    const min = Math.min(0, ...values);
    const max = Math.max(0, ...values);
    const span = max > min ? max - min : 1;
    const count = Math.max(1, ...series.map((item) => item.values.length));
    const x = (index) => pad + (count > 1 ? index / (count - 1) : 0.5) * (width - 2 * pad);
    const y = (value) => height - pad - ((value - min) / span) * (height - 2 * pad);
    const node = svg("svg", { class: "signed-series", viewBox: "0 0 " + width + " " + height, width: "100%", height: height, preserveAspectRatio: "none", });
    node.appendChild(svg("line", { class: "zero-axis", x1: 0, y1: y(0), x2: width, y2: y(0), stroke: "var(--line)", "stroke-width": 1, }));
    series.forEach((item, seriesIndex) => {
      const color = item.color || ["var(--acc)", "var(--amber)", "var(--ok)", "var(--err)"][seriesIndex % 4];
      let path = "";
      item.values.forEach((value, index) => {
        path += (index ? "L" : "M") + x(index).toFixed(1) + " " + y(value).toFixed(1) + " ";
      });
      if (path) node.appendChild(svg("path", {
        d: path.trim(), fill: "none", stroke: color, "stroke-width": 1.5,
      }));
      item.values.forEach((value, index) => {
        const dot = svg("circle", { cx: x(index), cy: y(value), r: 2, fill: color, "data-value": String(value), });
        const title = document.createElementNS(NS, "title");
        title.textContent = item.label + " [" + index + "] = " + value;
        dot.appendChild(title);
        node.appendChild(dot);
      });
    });
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

  // shortLabel: truncate a tick label so axis ticks stay legible.
  function shortLabel(text, keep) {
    text = String(text == null ? "" : text);
    keep = keep || 7;
    return text.length > keep ? text.slice(0, keep - 1) + "…" : text;
  }

  // colorScaleLegend: the value→color key for a heatmap. Reads the SAME ramp()
  // the cells use (sampled as gradient stops) with min/mid/max ticks, so a reader
  // can map color to magnitude instead of hovering every cell.
  function colorScaleLegend(min, max, opts) {
    opts = opts || {};
    const width = 200, barH = 10, height = 30, gid = "hm-grad-" + (legendSeq++);
    const node = svg("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height });
    const defs = svg("defs", {});
    const grad = svg("linearGradient", { id: gid, x1: "0", y1: "0", x2: "1", y2: "0" });
    for (const t of [0, 0.25, 0.5, 0.75, 1]) grad.appendChild(svg("stop", { offset: (t * 100) + "%", "stop-color": ramp(t) }));
    defs.appendChild(grad);
    node.appendChild(defs);
    node.appendChild(svg("rect", { x: 0, y: 0, width: width, height: barH, rx: 2, fill: "url(#" + gid + ")" }));
    const fmt = (v) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01)) ? v.toExponential(1) : v.toFixed(2);
    [[0, min, "start"], [width / 2, (min + max) / 2, "middle"], [width, max, "end"]].forEach(([x, v, anchor]) => {
      const label = svg("text", { x: x, y: barH + 15, "font-size": 10, fill: "var(--dim)", "text-anchor": anchor });
      label.textContent = (opts.label ? "" : "") + fmt(v);
      node.appendChild(label);
    });
    return node;
  }

  // heatmap: an N×N matrix as a grid of colored cells; color encodes magnitude in [min, max] via ramp().
  // Returns the grid plus a color-scale legend; optional row/col labels add axis ticks when the matrix
  // is small enough to stay legible. Values are shown as-is, no smoothing.
  function heatmap(matrix, opts) {
    opts = opts || {};
    const n = matrix.length;
    const max = opts.max != null ? opts.max : Math.max(1e-9, ...matrix.flat());
    const min = opts.min != null ? opts.min : 0;
    const span = max - min > 1e-12 ? max - min : 1;
    const cell = Math.max(6, Math.min(22, Math.floor(420 / Math.max(1, n))));
    const rowLabels = opts.rowLabels || opts.labels;
    const colLabels = opts.colLabels || opts.labels;
    // Ticks only when they will not collide (small n and labels provided).
    const showTicks = n <= 24 && cell >= 12;
    const marginLeft = showTicks && rowLabels ? 58 : 0;
    const marginTop = showTicks && colLabels ? 52 : 0;
    const grid = n * cell;
    const w = marginLeft + grid, h = marginTop + grid;
    const node = svg("svg", {
      viewBox: "0 0 " + w + " " + h,
      width: Math.min(w, 480), height: Math.min(h, 480),
      style: "max-width:100%",
    });
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const value = matrix[i][j];
        const rect = svg("rect", { x: marginLeft + j * cell, y: marginTop + i * cell, width: cell, height: cell, fill: ramp((value - min) / span) });
        const title = document.createElementNS(NS, "title");
        const rl = rowLabels ? " " + rowLabels[i] : "", cl = colLabels ? " " + colLabels[j] : "";
        title.textContent = "(" + i + rl + ", " + j + cl + ") = " + value.toFixed(4);
        rect.appendChild(title);
        node.appendChild(rect);
      }
    }
    if (showTicks && rowLabels) {
      for (let i = 0; i < n; i++) {
        const t = svg("text", { x: marginLeft - 5, y: marginTop + i * cell + cell / 2 + 3, "font-size": 10, fill: "var(--dim)", "text-anchor": "end" });
        t.textContent = shortLabel(rowLabels[i]);
        node.appendChild(t);
      }
    }
    if (showTicks && colLabels) {
      for (let j = 0; j < n; j++) {
        const x = marginLeft + j * cell + cell / 2;
        const t = svg("text", { x: x, y: marginTop - 5, "font-size": 10, fill: "var(--dim)", "text-anchor": "start", transform: "rotate(-55 " + x + " " + (marginTop - 5) + ")" });
        t.textContent = shortLabel(colLabels[j]);
        node.appendChild(t);
      }
    }
    if (opts.legend === false) return node;
    const wrap = document.createElement("div");
    wrap.appendChild(node);
    wrap.appendChild(colorScaleLegend(min, max, opts));
    return wrap;
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
  window.overgo.viz = { sparkline, signedSeries, probBars, heatmap, graph };
})();
