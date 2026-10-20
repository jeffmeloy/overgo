/* Scorecard: what a user feels of each activated model, per model and task, and what got better since the
   previous published scorecard. A cell that no record measures reads as one quiet dash, its reason beside it. */
(function () {
  "use strict";
  const columns = [["first-output", "First output"], ["throughput", "Throughput"], ["held-out-quality", "Quality"], ["failure-rate", "Failures"],
    ["load-time", "Load"], ["peak-device-memory", "Peak memory"], ["setup-steps", "Setup steps"]];
  const columnLabels = new Map(columns);
  const qualityColumn = "held-out-quality";

  window.overgo.registerTab({
    id: "scorecard",
    mount(panel, overgo) {
      const { el, fmt } = overgo;
      const unitFormats = { bytes: (value) => fmt.bytes(value), ms: (value) => fmt.elapsed(value), "tok/s": (value) => value.toFixed(1) + " tok/s" };
      const metricFormats = { "failure-rate": (value) => (100 * value).toFixed(1) + "%", "setup-steps": (value) => String(value) };
      const measure = (metric, unit, value) => (unitFormats[unit] || metricFormats[metric] || ((number) => number.toFixed(3)))(Number(value));

      // A gap is one quiet dash; why it is a gap waits in its tooltip and for assistive technology.
      const gap = (reason) => el("span", { class: "scorecard-gap", title: reason }, "—", el("span", { class: "sr-only", text: "not measured: " + reason }));
      // A held-out quality names what it measures and which way it improves; a trained model's reads against its base.
      const direction = { maximize: "higher is better", minimize: "lower is better" };
      function qualityCell(cell) {
        const sign = cell.direction === "maximize" ? 1 : cell.direction === "minimize" ? -1 : 0;
        const advantage = cell.base == null ? 0 : sign * (cell.value - cell.base);
        const verdict = advantage > 0 ? ["ok", "beats base"] : advantage < 0 ? ["caution", "below base"] : ["", "level with base"];
        return el("span", { title: [cell.label, direction[cell.direction]].filter(Boolean).join(" · ") },
          measure(cell.metric, cell.unit, cell.value), cell.measure ? el("span", { class: "note", text: " " + cell.measure }) : null,
          cell.base == null || !sign ? null : el("div", {}, el("span", { class: "tag " + verdict[0], title: "the base model scored " + measure(cell.metric, cell.unit, cell.base), text: verdict[1] })));
      }
      const scorecardCell = (cell) => cell.reason ? gap(cell.reason)
        : cell.metric === qualityColumn ? qualityCell(cell) : el("span", { class: "nowrap", title: cell.label || "", text: measure(cell.metric, cell.unit, cell.value) });
      // A chat model's first output at a longer prompt sits under its own, named by the prompt's tokens.
      const promptSize = (tokens) => tokens % 1024 ? tokens + " tokens" : tokens / 1024 + "K tokens";
      const lengthLine = (cell) => el("div", { class: "note nowrap", title: cell.label || "", text: measure(cell.metric, cell.unit, cell.value) + " at " + promptSize(cell.prompt) });
      function firstOutputCell(cell, lengths) {
        const shown = scorecardCell(cell);
        return lengths.length ? el("span", {}, shown, ...lengths.map(lengthLine)) : shown;
      }

      // renderHistory: every change from the previous published scorecard to the newest, grouped by how it moved
      // and whether a user would call it better, targeted or not.
      const historyHost = el("div", { class: "scorecard-history" });
      function renderHistory(history) {
        if (!history) {
          historyHost.replaceChildren(el("p", { class: "note", text: "The unattended loop publishes a scorecard each time it empties its plan; what moved shows here once two are published." }));
          return;
        }
        if (history.unavailable) {
          historyHost.replaceChildren(el("p", { class: "note", text: "What moved between scorecards could not be read: " + history.unavailable }));
          return;
        }
        const moves = history.moves || [];
        const where = (move) => move.name + " " + move.task + (move.metric ? ", " + (columnLabels.get(move.metric) || move.metric) : "") +
          (move.prompt ? " at " + promptSize(move.prompt) : "");
        const value = (move, number) => measure(move.metric, move.unit, number);
        const change = (move) => where(move) + ": " + value(move, move.before) + " to " + value(move, move.after);
        // A cell measured another way than the previous scorecard's has no comparison; the technical reason waits in the tooltip.
        const groups = [
          ["better", "Better", change],
          ["worse", "Worse", change],
          ["changed", "Changed, with no better or worse", change],
          ["measured", "Newly measured", (move) => where(move) + ": " + value(move, move.after)],
          ["unmeasured", "No longer measured", (move) => where(move) + ": " + move.reason],
          ["unknown", "Measured differently, so not compared", where],
          ["added", "New", where],
          ["removed", "Gone", where],
        ].map(([kind, title, line]) => [kind, title, moves.filter((move) => move.kind === kind).map((move) => [line(move), move.kind === "unknown" ? move.reason : ""])]).filter(([, , lines]) => lines.length);
        const count = (kind) => (groups.find(([name]) => name === kind) || [, , []])[2].length;
        historyHost.replaceChildren(
          el("p", { class: "note", text: "Since the scorecard of commit " + history.previous.slice(0, 12) + ", the one of " + history.current.slice(0, 12) + ": " +
            (moves.length ? count("better") + " better, " + count("worse") + " worse. Every change is listed, whether or not work aimed at it." : "nothing moved.") }),
          ...groups.map(([kind, title, lines]) => el("div", { class: "scorecard-moves", "data-kind": kind },
            el("h4", { text: title }), el("ul", {}, ...lines.map(([text, reason]) => el("li", { title: reason, text }))))));
      }

      const body = el("tbody"), note = el("div", { class: "note" });
      const read = overgo.read(note, (signal) => overgo.api.get("/scorecard", { signal }), (answer) => {
        renderHistory(answer.history);
        const rows = answer.rows || [];
        note.textContent = rows.length ? "What a user feels of each activated model. A dash means no record measures it yet; hover it for why." : "No activated model to score yet.";
        body.replaceChildren(...rows.map((row) => {
          const cells = new Map((row.cells || []).map((cell) => [cell.metric, cell]));
          const shown = columns.map(([metric]) => !cells.has(metric) ? gap("the scorecard reads no " + metric)
            : metric === "first-output" ? firstOutputCell(cells.get(metric), row.lengths || []) : scorecardCell(cells.get(metric)));
          const line = overgo.tableRow([el("span", { text: row.name }), el("span", { text: row.task }), ...shown]);
          ["Model", "Task", ...columns.map(([, label]) => label)].forEach((label, index) => { line.children[index].dataset.label = label; });
          // A phone stacks a row under its headers and shows only what was measured.
          shown.forEach((cell, index) => line.children[index + 2].classList.toggle("gap", cell.classList.contains("scorecard-gap")));
          return line;
        }));
      }, { loading: "Loading the scorecard…" });

      panel.append(
        el("div", { class: "row" }, el("div", { class: "section-title", text: "What got better" }), el("button", { class: "link-button", text: "Refresh", onclick: () => read() })),
        historyHost,
        el("div", { class: "section-title", text: "Scorecard" }), note,
        el("table", { class: "grid stack" }, el("thead", null, overgo.headerRow(["Model", "Task", ...columns.map(([, label]) => label)])), body));
      return read();
    },
  });
})();
