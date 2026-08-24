/* Library: model and dataset discovery. Three surfaces in one panel: the
   local catalog exactly as the store proves it servable, Hugging Face search
   over models and datasets, and verified download jobs with live progress.
   Downloads land under the server's configured root through the hub client's
   digest-verified path. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "library",
    label: "Library",
    section: "library",
    mount(panel, overgo) {
      const { el, clear, fmt } = overgo;
      clear(panel);

      // ---- local catalog ----
      const catalogBody = el("tbody");
      const catalogNote = el("div", { class: "note", text: "loading catalog…" });
      async function refreshCatalog() {
        try {
          const catalog = await overgo.api.get("/catalog/models");
          overgo.clear(catalogBody);
          const models = catalog.models || [];
          const coverage = catalog.coverage || {};
          const parts = [];
          if (models.length) parts.push(models.length + " activated model(s)");
          const profiles = coverage.profiles || {};
          if (profiles.registered) parts.push("profiles " + profiles.published + "/" + profiles.registered);
          const datasets = coverage.datasets || {};
          if (datasets.registered) parts.push("datasets " + datasets.available + "/" + datasets.registered + " on disk");
          if (catalog.truncated) parts.push("listing truncated — not every activation is shown");
          catalogNote.textContent = parts.length ? parts.join(" · ") :
            "No activated models yet — download one below, then activate it with cmd/reverify.";
          for (const entry of models) {
            const capabilities = el("td");
            for (const capability of entry.capabilities || []) {
              const label = capability.stale ? capability.task + " · stale" :
                capability.task + (capability.tier ? " · " + capability.tier : "");
              capabilities.appendChild(el("span", {
                class: "tag", style: "margin-right:4px",
                title: capability.stale || capability.recipe, text: label,
              }));
            }
            catalogBody.appendChild(el("tr", null,
              el("td", { class: "mono", text: fmt.shortID(entry.model) }),
              el("td", { text: (entry.location || "").split(/[\\/]/).pop() }),
              capabilities,
              el("td", null, entry.present ? "" : el("span", { class: "tag", text: "missing bytes" }))));
          }
        } catch (err) {
          catalogNote.textContent = overgo.friendlyError(err);
        }
      }

      // ---- hub search ----
      const query = el("input", { class: "text", placeholder: "search the Hugging Face hub" });
      const kind = el("select", { class: "text", style: "width:130px" },
        el("option", { value: "models", text: "models" }),
        el("option", { value: "datasets", text: "datasets" }));
      const searchNote = el("span", { class: "note" });
      const resultsBody = el("tbody");
      const searchButton = el("button", { class: "btn" }, "search");
      const searchCancel = el("button", { class: "btn alt", style: "display:none" }, "cancel");
      const runSearch = overgo.runner(searchButton, searchCancel, {
        onError: (err) => { searchNote.textContent = overgo.friendlyError(err); },
      });
      function search() {
        return runSearch(async (signal) => {
          searchNote.textContent = "searching…";
          const path = "/hub/search?kind=" + kind.value + "&q=" + encodeURIComponent(query.value);
          const found = await overgo.api.get(path, { signal });
          overgo.clear(resultsBody);
          const results = found.results || [];
          searchNote.textContent = results.length ? results.length + " result(s)" : "no results";
          for (const listing of results) {
            const download = el("button", { class: "btn alt", onclick: () => startDownload(listing.id) }, "download");
            resultsBody.appendChild(el("tr", null,
              el("td", { class: "mono", text: listing.id }),
              el("td", { class: "mono", text: fmt.compact(listing.downloads || 0) }),
              el("td", { class: "mono", text: fmt.compact(listing.likes || 0) }),
              el("td", null, listing.gated ? el("span", { class: "tag control", text: "gated" }) : download)));
          }
        });
      }
      query.addEventListener("keydown", (event) => { if (event.key === "Enter") search(); });
      searchButton.addEventListener("click", search);

      // ---- downloads ----
      const jobsBody = el("tbody");
      const jobsNote = el("div", { class: "note" });
      async function startDownload(repository) {
        jobsNote.textContent = "starting " + repository + "…";
        try {
          await overgo.api.post("/hub/downloads", { kind: kind.value, repository: repository, revision: "", directory: "" });
          jobsNote.textContent = "";
        } catch (err) {
          jobsNote.textContent = overgo.friendlyError(err);
        }
      }
      const jobPoller = overgo.poller(async (signal) => {
        const listed = await overgo.api.get("/hub/downloads", { signal });
        overgo.clear(jobsBody);
        for (const job of listed.downloads || []) {
          const progress = job.total > 0 ? Math.floor(job.received * 100 / job.total) + "%" : "";
          const state = job.state === "failed" ? el("span", { class: "tag control", text: "failed" }) :
            job.state === "succeeded" ? el("span", { class: "tag user_defined", text: "done" }) :
              el("span", { class: "tag byte", text: progress || "running" });
          jobsBody.appendChild(el("tr", null,
            el("td", { class: "mono", text: job.repository }),
            el("td", null, state),
            el("td", { class: "mono", text: job.file || (job.state === "failed" ? job.error : "") }),
            el("td", { class: "mono", text: job.total > 0 ? fmt.bytes(job.received) + " / " + fmt.bytes(job.total) : (job.files ? job.files + " files" : "") })));
        }
      }, 1000);

      panel.append(
        el("div", { class: "section-title", text: "Local models" }),
        catalogNote,
        el("table", { class: "grid" },
          el("thead", null, el("tr", null,
            el("th", { text: "model" }), el("th", { text: "file" }), el("th", { text: "capabilities" }), el("th", { text: "" }))),
          catalogBody),
        el("div", { class: "section-title", text: "Hugging Face" }),
        el("div", { class: "row" }, kind, query, searchButton, searchCancel, searchNote),
        el("table", { class: "grid" },
          el("thead", null, el("tr", null,
            el("th", { text: "repository" }), el("th", { text: "downloads" }), el("th", { text: "likes" }), el("th", { text: "" }))),
          resultsBody),
        el("div", { class: "section-title", text: "Downloads" }),
        jobsNote,
        el("table", { class: "grid" },
          el("thead", null, el("tr", null,
            el("th", { text: "repository" }), el("th", { text: "state" }), el("th", { text: "file" }), el("th", { text: "progress" }))),
          jobsBody));

      refreshCatalog();
      this.onActivate = () => jobPoller.start();
      this.onDeactivate = () => jobPoller.stop();
      jobPoller.start();
    },
  });
})();
