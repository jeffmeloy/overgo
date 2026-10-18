/* Requests: every API request the server answered, newest first, from the server's bounded ring
   (/requests): a header of what the ring shows and the requests in flight or failed on their own. */
(function () {
  "use strict";
  // The ring is read again this often while the tab shows; the tab's own reads never enter it.
  const refreshMS = 2000;
  // A status from here up is a failure; a request still in flight has none yet.
  const failedFrom = 400;
  const filters = [["all", "All requests"], ["in-flight", "In flight"], ["failed", "Failed"]];
  const failed = (item) => !item.in_flight && item.status >= failedFrom;

  // totals: the ring's requests as one line, the error rate over those that ended, the slowest of them,
  // and the completion tokens per second the generating ones made over the time they took.
  function totals(answer, fmt) {
    const ended = answer.requests.filter((item) => !item.in_flight);
    const generating = ended.filter((item) => item.completion_tokens);
    const seconds = generating.reduce((sum, item) => sum + item.duration_ms, 0) / 1000;
    const slowest = ended.reduce((worst, item) => (!worst || item.duration_ms > worst.duration_ms ? item : worst), null);
    const parts = [answer.total + " requests since the server started", answer.requests.filter((item) => item.in_flight).length + " in flight"];
    if (ended.length) parts.push(Math.round(100 * ended.filter(failed).length / ended.length) + "% failed of the last " + ended.length);
    if (slowest) parts.push("slowest " + slowest.method + " " + slowest.route + " " + fmt.elapsed(slowest.duration_ms));
    if (seconds) parts.push((generating.reduce((sum, item) => sum + item.completion_tokens, 0) / seconds).toFixed(1) + " tokens/s");
    return parts.join(" · ");
  }

  // tokens: what a request's generations read and wrote, with the prompt tokens the cache already held and their share.
  function tokens(item) {
    if (!item.prompt_tokens && !item.completion_tokens) return "";
    const reused = item.cached_tokens ? " (" + item.cached_tokens + " reused, " + Math.round(100 * item.cached_tokens / item.prompt_tokens) + "%)" : "";
    return item.prompt_tokens + " in" + reused + " · " + item.completion_tokens + " out";
  }

  // speeds: prefill reads the prompt tokens the cache did not hold over the time they took, decode the output tokens
  // over the time after; each shows only when the server measured it.
  function speeds(item) {
    const rate = (count, ms) => (count / (ms / 1000)).toFixed(1) + " tok/s";
    const parts = [];
    if (item.prefill_ms) parts.push("prefill " + rate(item.prompt_tokens - (item.cached_tokens || 0), item.prefill_ms));
    if (item.decode_ms && item.completion_tokens) parts.push("decode " + rate(item.completion_tokens, item.decode_ms));
    return parts.join(" · ");
  }

  // waited: what a request spent before it generated, waiting for a place to run (after a stop, the slot before it
  // unwinding) and preparing its prompt, each shown only when the server measured it.
  function waited(item, fmt) {
    const parts = [];
    if (item.admission_ms) parts.push("place " + fmt.elapsed(item.admission_ms));
    if (item.preparation_ms) parts.push("prompt " + fmt.elapsed(item.preparation_ms));
    return parts.join(" · ");
  }

  window.overgo.registerTab({
    id: "requests",
    async mount(panel, overgo) {
      const { api, el, fmt } = overgo;
      const summary = el("div", { class: "note", role: "group", "aria-label": "Request totals" });
      const show = el("select", { class: "text", "aria-label": "Show" }, ...filters.map(([value, text]) => el("option", { value, text })));
      const listHost = el("div");
      panel.replaceChildren(el("div", { class: "section-title", text: "Requests" }), el("label", { class: "control" }, el("span", { text: "Show" }), show), summary, listHost);

      let latest = null;
      function draw() {
        if (!latest) return;
        summary.textContent = totals(latest, fmt);
        const shown = latest.requests.filter((item) => show.value === "all" || (show.value === "in-flight" ? item.in_flight : failed(item)));
        listHost.replaceChildren(overgo.table(["time", "method", "route", "status", "duration", "tokens", "speed", "waited"], shown.map((item) => [
          new Date(item.started_ms).toLocaleTimeString(),
          item.method,
          item.route,
          item.in_flight ? el("span", { class: "tag", text: "in flight" }) : failed(item) ? el("span", { class: "tag caution", text: String(item.status) }) : String(item.status),
          fmt.elapsed(item.duration_ms),
          tokens(item),
          speeds(item),
          waited(item, fmt),
        ]), "request-grid", "No requests match."));
      }
      show.addEventListener("input", draw);
      // The placeholder is the first read's alone: a read again every few seconds must not blank the table it replaces.
      // A payload that is not a list is refused before it is kept, so a later change of filter draws the last good one.
      listHost.replaceChildren(el("div", { class: "note", text: "Loading…" }));
      const refresh = overgo.read(listHost, (signal) => api.get("/requests", { signal }), (answer) => {
        if (!answer || !Array.isArray(answer.requests)) throw new Error("the server's request log was not a list");
        latest = answer;
        draw();
      }, { loading: null });
      // The tab is a view its owner mounts, so it ends with the owner's next choice, and it reads only while it shows.
      const visible = () => !document.hidden && (typeof panel.checkVisibility !== "function" || panel.checkVisibility());
      const timer = setInterval(() => { if (visible()) refresh(); }, refreshMS);
      overgo.onRelease(() => clearInterval(timer));
      await refresh();
    },
  });
})();
