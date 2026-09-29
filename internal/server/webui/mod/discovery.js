/* Library: model and dataset discovery. Three surfaces in one panel: the
   local catalog exactly as the store proves it servable, Hugging Face search
   over models and datasets, and verified download jobs with live progress.
   Downloads land under the server's configured root through the hub client's
   digest-verified path. */
(function () {
  "use strict";
  window.overgo.registerTab({
    id: "library",
    mount(panel, overgo) {
      const { el, fmt } = overgo;

      // ---- local catalog: the newest read wins ----
      const catalogBody = el("tbody"), catalogNote = el("div", { class: "note" });
      const refreshCatalog = overgo.read(catalogNote, (signal) => overgo.api.get("/catalog/models", { signal }), (catalog) => {
        overgo.clear(catalogBody);
        const models = catalog.models || [];
        const { profiles = {}, datasets = {} } = catalog.coverage || {};
        const parts = [models.length && models.length + (models.length === 1 ? " model is ready to serve" : " models are ready to serve"),
          profiles.registered && profiles.published + " of " + profiles.registered + " architecture profiles published",
          datasets.registered && datasets.available + " of " + datasets.registered + " datasets on disk", catalog.truncated && "the listing is truncated, so not every model is shown"].filter(Boolean);
        catalogNote.textContent = parts.length ? parts.join(" · ") :
          "No model is ready to serve yet. Search Hugging Face or add a model from disk below, then validate it.";
        // Verified is the norm the listing states once; a tag names only a state that differs from it.
        const standing = (capability) => {
          if (capability.stale) return " · stale";
          return capability.tier && capability.tier !== "verified" ? " · " + capability.tier : "";
        };
        for (const entry of models) {
          const capabilities = el("span", {}, ...(entry.capabilities || []).map((capability) => el("span", {
            class: "tag mr-4", title: capability.stale || capability.recipe, text: capability.task + standing(capability) })));
          // A hosted model retires from its row with the section's reason; the catalog relists without it.
          const retire = (entry.location || "").startsWith("remote://") ? el("button", { class: "btn alt", text: "Retire", onclick: async () => {
            try { await overgo.api.post("/library/providers/retire", { location: entry.location, reason: providerReason.value.trim() || "retired from the Library tab" }); refreshCatalog(); } catch (err) { catalogNote.replaceChildren(overgo.failure(err)); }
          } }) : "";
          // A model reads by its file and what it does; its identity follows for reference.
          const line = overgo.tableRow([el("span", { text: (entry.location || "").split(/[\\/]/).pop() }), capabilities,
            el("span", { class: "note mono", title: entry.model, text: fmt.shortID(entry.model) }), entry.present ? retire : el("span", { class: "tag", text: "missing bytes" })]);
          ["Model", "Capabilities", "Identity", ""].forEach((label, index) => { line.children[index].dataset.label = label; });
          catalogBody.appendChild(line);
        }
      }, { loading: "Loading the catalog…" });

      // ---- hub search ----
      const query = el("input", { "aria-label": "Search the Hugging Face hub", class: "text", placeholder: "Search the Hugging Face hub" });
      const kind = el("select", { class: "text w-130", "aria-label": "search kind" }, el("option", { value: "models", text: "models" }), el("option", { value: "datasets", text: "datasets" }));
      const searchNote = el("span", { class: "note" });
      const resultsBody = el("tbody");
      const searchButton = el("button", { class: "btn" }, "Search");
      const searchCancel = el("button", { class: "btn alt", hidden: true }, "Cancel");
      const runSearch = overgo.runner(searchButton, searchCancel, {
        onError: (err) => { searchNote.textContent = overgo.friendlyError(err); },
      });
      function search() {
        return runSearch(async (signal) => {
          searchNote.textContent = "Searching…";
          const path = "/hub/search?kind=" + kind.value + "&q=" + encodeURIComponent(query.value);
          const found = await overgo.api.get(path, { signal });
          overgo.clear(resultsBody);
          const results = found.results || [];
          const counted = results.length === 1 ? "1 result" : results.length + " results";
          searchNote.textContent = results.length ? counted : "No results";
          for (const listing of results) {
            const download = el("button", { class: "btn alt", onclick: () => startDownload(listing.id) }, "Add");
            resultsBody.appendChild(overgo.tableRow([listing.id, fmt.compact(listing.downloads || 0), fmt.compact(listing.likes || 0), listing.gated ? el("span", { class: "tag control", text: "gated" }) : download]));
          }
        });
      }
      query.addEventListener("keydown", (event) => { if (event.key === "Enter") search(); }); searchButton.addEventListener("click", search);

      // ---- downloads ----
      const jobsBody = el("tbody");
      const localBody = el("tbody");
      const jobsNote = el("div", { class: "note" });
      // Adding a hub model downloads it, then registers and validates it once the download lands.
      const chained = new Set();
      async function startDownload(repository) {
        jobsNote.textContent = "starting " + repository + "…";
        chained.add(repository);
        try { await overgo.api.post("/hub/downloads", { kind: kind.value, repository: repository, revision: "", directory: "" }); jobsNote.textContent = ""; }
        catch (err) { chained.delete(repository); jobsNote.textContent = overgo.friendlyError(err); }
      }
      // ---- the lifecycle after a download: register, then validate; a model validates as an operation the
      // strip shows, a dataset by preview. Adding runs both; each button shows its step and runs it again ----
      const lifecycles = new Map();
      function stageOf(job) {
        if (!lifecycles.has(job.id)) lifecycles.set(job.id, { host: el("span"), registered: null, pending: false });
        return lifecycles.get(job.id);
      }
      async function step(job, run) {
        const stage = stageOf(job);
        if (stage.pending) return false;
        stage.pending = true; redraw(job);
        try { await run(stage, job.repository.split("/").pop()); return true; }
        catch (err) { stage.host.replaceChildren(overgo.failure(err)); return false; }
        finally { stage.pending = false; redraw(job); }
      }
      // registration: a model by its file, a dataset file importing a common chat format, or a downloaded directory.
      function registration(job, name) {
        if (job.kind !== "datasets") return { kind: "model", path: job.destination, projector: job.projector || "" };
        return job.source ? { kind: "dataset", name, path: job.source } : { kind: "dataset", name, directory: job.destination };
      }
      const register = (job) => step(job, async (stage, name) => {
        if (stage.registered) return;
        stage.registered = await overgo.api.post("/library/register", registration(job, name));
        const imported = stage.imported = (stage.registered.formats || []).length ? " · " + stage.registered.files + " conversations from " + stage.registered.formats.join(", ") : "";
        stage.host.replaceChildren(el("span", { class: "note" }, "registered ", overgo.artifactLink(stage.registered.recipe || stage.registered.dataset),
          stage.registered.projector ? " with projector " + (stage.registered.media || []).join("/") : "", imported));
        refreshCatalog();
      });
      const validate = (job) => step(job, async (stage, name) => {
        if (job.kind === "datasets") {
          const preview = await overgo.api.post("/datasets/preview", { name, position: 0, limit: 1 });
          stage.host.replaceChildren(el("span", { class: "note", text: "validated: " + preview.examples.length + " example previewed" + (stage.imported || "") }));
          return;
        }
        const admitted = await overgo.api.post("/library/validate", { path: stage.registered.path, projector: stage.registered.projector || "" });
        stage.host.replaceChildren(el("span", { class: "note" }, "validating in operation ", overgo.artifactLink(admitted.operation), admitted.projector ? " · projector" : ""));
        if (!overgo.signal.aborted && panel.classList.contains('active')) overgo.showOperation(admitted.operation);
      });
      async function add(job) { if (await register(job) && stageOf(job).registered) await validate(job); }
      function redraw(job) { const stage = stageOf(job); if (!overgo.signal.aborted && stage.cell) stage.cell.replaceWith(lifecycle(job)); }
      function lifecycle(job) {
        const stage = stageOf(job);
        const cell = el("span", { class: "row" });
        stage.cell = cell;
        if (job.state !== "succeeded") return cell;
        cell.append(
          el("button", { class: "btn alt", text: stage.registered ? "registered" : "register", disabled: !!stage.registered || stage.pending, onclick: () => register(job) }),
          el("button", { class: "btn", text: "validate", disabled: !stage.registered || stage.pending, onclick: () => validate(job) }),
          stage.host);
        return cell;
      }
      // Download jobs arrive on the runtime stream: every job at connect, then again whenever one reads
      // differently. A server with no download surface (the cold page) says why; the tab remounts when a model serves.
      let streamRefusal = false;
      overgo.subscribe((event, data) => {
        if (event === "stream.idle" || event === "stream.error") { jobsNote.textContent = overgo.friendlyError(data); streamRefusal = true; }
        if (event === "stream.ready" && streamRefusal) { jobsNote.textContent = ""; streamRefusal = false; }
        if (event !== "hub.downloads") return;
        jobsBody.replaceChildren(...(data || []).map((job) => {
          const progress = job.total > 0 ? Math.floor(job.received * 100 / job.total) + "%" : "";
          const state = job.state === "failed" ? el("span", { class: "tag control", text: "failed" }) : job.state === "succeeded" ? el("span", { class: "tag user_defined", text: "done" }) : el("span", { class: "tag byte", text: progress || "running" });
          return overgo.tableRow([job.repository, state, job.file || (job.state === "failed" ? job.error : ""), job.total > 0 ? fmt.bytes(job.received) + " / " + fmt.bytes(job.total) : (job.files ? job.files + " files" : ""), lifecycle(job)]);
        }));
        for (const job of data || []) if (job.state === "succeeded" && chained.delete(job.repository)) add(job);
      }, { whileShown: true });

      // ---- the user scorecard: per model and task, what a user feels; an unmeasured cell names why ----
      const scorecardColumns = [["first-output", "First output"], ["throughput", "Throughput"], ["held-out-quality", "Quality"], ["failure-rate", "Failures"],
        ["load-time", "Load"], ["peak-device-memory", "Peak memory"], ["setup-steps", "Setup steps"]];
      const unitFormats = { bytes: (value) => fmt.bytes(value), ms: (value) => value.toFixed(1) + " ms", "tok/s": (value) => value.toFixed(1) + " tok/s" };
      const metricFormats = { "failure-rate": (value) => (100 * value).toFixed(1) + "%", "setup-steps": (value) => String(value) };
      const scorecardCell = (cell) => cell.reason ? el("span", { class: "scorecard-gap", title: cell.reason, text: "not measured" })
        : el("span", { title: cell.label || "", text: (unitFormats[cell.unit] || metricFormats[cell.metric] || ((value) => value.toFixed(3)))(Number(cell.value)) });
      const scorecardBody = el("tbody"), scorecardNote = el("div", { class: "note" });
      const scorecard = el("section", { class: "scorecard", role: "region", "aria-label": "scorecard", hidden: true }, el("div", { class: "section-title", text: "Scorecard" }), scorecardNote,
        el("table", { class: "grid stack" }, el("thead", null, overgo.headerRow(["Model", "Task", ...scorecardColumns.map(([, label]) => label)])), scorecardBody));
      async function refreshScorecard() {
        let read;
        try { read = await overgo.api.get("/scorecard", { signal: overgo.signal }); }
        catch (err) {
          // A server without the launcher's scorecard reader has no scorecard to show.
          if (overgo.signal.aborted || err.status === 404) return;
          scorecard.hidden = false;
          scorecardNote.replaceChildren(overgo.failure(err));
          return;
        }
        scorecard.hidden = false;
        const rows = read.rows || [];
        scorecardNote.textContent = rows.length ? "What a user feels of each activated model; hover a cell for what measured it." : "No activated model to score yet.";
        scorecardBody.replaceChildren(...rows.map((row) => {
          const cells = new Map((row.cells || []).map((cell) => [cell.metric, cell]));
          const line = overgo.tableRow([el("span", { text: row.name }), el("span", { text: row.task }),
            ...scorecardColumns.map(([metric]) => cells.has(metric) ? scorecardCell(cells.get(metric)) : el("span", { class: "scorecard-gap", text: "not measured" }))]);
          ["Model", "Task", ...scorecardColumns.map(([, label]) => label)].forEach((label, index) => { line.children[index].dataset.label = label; });
          return line;
        }));
      }
      refreshScorecard();

      const downloadsTitle = el("div", { class: "section-title", text: "Downloads" });
      panel.append(
        el("div", { class: "row" }, el("div", { class: "section-title", text: "Local models" }), el('button', { class: 'link-button', text: 'Refresh catalog', onclick: () => { refreshCatalog(); refreshScorecard(); } })),
        catalogNote,
        el("table", { class: "grid stack" }, el("thead", null, overgo.headerRow(["Model", "Capabilities", "Identity", ""])), catalogBody),
        scorecard,
        el("div", { class: "section-title", text: "Hugging Face" }),
        el("div", { class: "row" }, kind, query, searchButton, searchCancel, searchNote),
        el("table", { class: "grid hide-empty" }, el("thead", null, overgo.headerRow(["repository", "downloads", "likes", ""])), resultsBody),
        downloadsTitle,
        jobsNote,
        el("table", { class: "grid hide-empty" }, el("thead", null, overgo.headerRow(["repository", "state", "file", "progress", "register · validate"])), localBody, jobsBody));

      // ---- a local model: a GGUF (or its directory) already on disk, with its projector, rides the same lifecycle ----
      const localModel = el("input", { "aria-label": "Model GGUF or directory on disk", class: "text", placeholder: "model GGUF or directory on disk" });
      const localProjector = el("input", { "aria-label": "Projector GGUF (optional)", class: "text", placeholder: "projector GGUF (optional)" });
      const localButton = el("button", { class: "btn alt", text: "Add a local model", onclick: () => {
        const path = localModel.value.trim(); if (!path) return;
        const job = { id: "local:" + path, kind: "models", state: "succeeded", repository: path, destination: path, projector: localProjector.value.trim() };
        localBody.replaceChildren(overgo.tableRow([path, el("span", { class: "tag user_defined", text: "local" }), job.projector, "", lifecycle(job)]));
        add(job);
      } });
      const fromDisk = [el("div", { class: "section-title", text: "Add a model from disk" }), el("div", { class: "row" }, localModel, localProjector, localButton)];

      // ---- a dataset file in alpaca, ShareGPT or OpenAI messages form imports as conversations, one per record ----
      const datasetSource = el("input", { "aria-label": "Chat dataset file on disk", class: "text", placeholder: "alpaca, ShareGPT or OpenAI messages file (.json or .jsonl)" });
      const datasetName = el("input", { "aria-label": "Dataset name", class: "text", placeholder: "dataset name" });
      const datasetButton = el("button", { class: "btn alt", text: "Import a dataset file", onclick: () => {
        const path = datasetSource.value.trim(), name = datasetName.value.trim(); if (!path || !name) return;
        const job = { id: "import:" + path, kind: "datasets", state: "succeeded", repository: name, source: path };
        localBody.replaceChildren(overgo.tableRow([name, el("span", { class: "tag user_defined", text: "import" }), path, "", lifecycle(job)]));
        add(job);
      } });
      // Each way a model or dataset comes in is its own section; Downloads then follows what they started.
      downloadsTitle.before(...fromDisk, el("div", { class: "section-title", text: "Import a chat dataset" }),
        el("div", { class: "row" }, datasetSource, datasetName, datasetButton),
        el("p", { class: "note", text: "Imported conversations preview under Datasets; the workbench trains preference and rollout datasets." }));

      // ---- a hosted provider: the declaration the CLI file carries, committed through the library route; the catalog relists ----
      const providerFields = ["name", "endpoint", "key variable", "model ids (comma-separated)", "context length (optional)"].map((placeholder) => el("input", { class: "text", placeholder, "aria-label": "provider " + placeholder }));
      const providerNote = el("span", { class: "note" });
      const providerListing = el("span", { class: "row" });
      const providerReason = el("input", { class: "text", placeholder: "retirement reason (the endpoint gone, the key withdrawn)", "aria-label": "retirement reason" });
      const providerValues = () => providerFields.map((field) => field.value.trim());
      const providerButton = el("button", { class: "btn alt", text: "Declare a hosted provider", onclick: async () => {
        const [name, endpoint, keyEnvironment, models, contextLength] = providerValues();
        try {
          const declared = await overgo.api.post("/library/register", { kind: "provider", name, endpoint, key_environment: keyEnvironment, models: models.split(",").map((id) => id.trim()).filter(Boolean), context_length: Number(contextLength) || 0 });
          providerNote.replaceChildren(...(declared.declared || []).map((item) => el("span", { class: "tag", title: item.recipe, text: item.location + (item.refusal ? " · " + item.refusal : "") })));
          refreshCatalog();
        } catch (err) { providerNote.textContent = overgo.friendlyError(err); }
      } });
      // The provider's own listing (its key gates it): a listed model picked here fills the model ids and its declared context length.
      const listButton = el("button", { class: "btn alt", text: "list the provider's models", onclick: async () => {
        const [, endpoint, keyEnvironment] = providerValues();
        try {
          const listed = await overgo.api.get("/library/providers/models?endpoint=" + encodeURIComponent(endpoint) + "&key_environment=" + encodeURIComponent(keyEnvironment));
          providerListing.replaceChildren(...(listed.models || []).map((model) => el("button", { class: "btn alt", text: model.id + (model.context_length ? " · " + model.context_length : ""), title: model.name || model.id, onclick: () => {
            providerFields[3].value = [...providerFields[3].value.split(",").map((id) => id.trim()).filter(Boolean), model.id].join(", "); if (model.context_length) providerFields[4].value = String(model.context_length);
          } })));
        } catch (err) { providerNote.textContent = overgo.friendlyError(err); }
      } });
      panel.append(el("div", { class: "section-title", text: "Hosted providers" }), el("div", { class: "row" }, ...providerFields, listButton, providerButton), providerListing, providerNote, el("div", { class: "row" }, providerReason));

      // Each showing reads the catalog again.
      overgo.whileActive(refreshCatalog);
    },
  });
})();
