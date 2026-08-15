/* Landing probe: poll /health and, once the server answers, redirect into the
   workbench. Kept as an external file (not inline) so the served pages can carry
   a strict Content-Security-Policy with script-src 'self' — no inline scripts. */
(function () {
  const $ = (id) => document.getElementById(id);
  let tries = 0, timer = null;

  window.probe = async function (manual) {
    tries++;
    $("probe-count").textContent = "probe #" + tries;
    if (manual) { $("dot").className = "dot scan"; $("probe-text").textContent = "probing…"; }
    try {
      const health = await fetch("/health", { signal: AbortSignal.timeout(2500) });
      if (!health.ok) throw new Error(String(health.status));
      let modelName = "";
      try {
        const models = await fetch("/models", { signal: AbortSignal.timeout(2500) });
        if (models.ok) {
          const body = await models.json();
          modelName = (body.models && body.models[0] && body.models[0].name) || "";
        }
      } catch (_) { /* /models is best-effort for the label */ }
      $("dot").className = "dot ok";
      $("probe-text").textContent = "server online" + (modelName ? " — " + modelName : "") + " — entering…";
      $("hint").style.display = "none";
      $("retry").style.display = "none";
      $("enter").style.display = "";
      clearInterval(timer);
      setTimeout(() => { location.href = "/app.html"; }, manual ? 400 : 900);
    } catch (e) {
      $("dot").className = "dot err";
      $("probe-text").textContent = "no server at " + location.origin;
      $("hint").style.display = "";
      $("retry").style.display = "";
    }
  };
  // Wired here rather than an inline onclick so the page needs no inline scripts
  // (strict CSP: script-src 'self').
  $("retry").addEventListener("click", () => probe(true));
  probe(false);
  timer = setInterval(() => probe(false), 4000);
})();
