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
          const { profiles = {}, datasets = {} } = catalog.coverage || {};
          const parts = [models.length && models.length + " activated model(s)", profiles.registered && "profiles " + profiles.published + "/" + profiles.registered,
            datasets.registered && "datasets " + datasets.available + "/" + datasets.registered + " on disk", catalog.truncated && "listing truncated — not every activation is shown"].filter(Boolean);
          catalogNote.textContent = parts.length ? parts.join(" · ") :
            "No activated models yet — download one below, then activate it with cmd/reverify.";
          for (const entry of models) {
            const capabilities = el("td", {}, ...(entry.capabilities || []).map((capability) => el("span", {
              class: "tag", style: "margin-right:4px", title: capability.stale || capability.recipe,
              text: capability.task + (capability.stale ? " · stale" : capability.tier ? " · " + capability.tier : "") })));
            catalogBody.appendChild(overgo.tableRow([fmt.shortID(entry.model), el("span", { text: (entry.location || "").split(/[\\/]/).pop() }), capabilities, entry.present ? "" : el("span", { class: "tag", text: "missing bytes" })]));
          }
        } catch (err) { catalogNote.textContent = overgo.friendlyError(err); }
      }

      // ---- hub search ----
      const query = el("input", { class: "text", placeholder: "search the Hugging Face hub" });
      const kind = el("select", { class: "text", style: "width:130px" }, el("option", { value: "models", text: "models" }), el("option", { value: "datasets", text: "datasets" }));
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
            resultsBody.appendChild(overgo.tableRow([listing.id, fmt.compact(listing.downloads || 0), fmt.compact(listing.likes || 0), listing.gated ? el("span", { class: "tag control", text: "gated" }) : download]));
          }
        });
      }
      query.addEventListener("keydown", (event) => { if (event.key === "Enter") search(); });
      searchButton.addEventListener("click", search);

      // ---- downloads ----
      const jobsBody = el("tbody");
      const localBody = el("tbody");
      const jobsNote = el("div", { class: "note" });
      async function startDownload(repository) {
        jobsNote.textContent = "starting " + repository + "…";
        try { await overgo.api.post("/hub/downloads", { kind: kind.value, repository: repository, revision: "", directory: "" }); jobsNote.textContent = ""; } catch (err) { jobsNote.textContent = overgo.friendlyError(err); }
      }
      // ---- the lifecycle after a download: register once the download succeeded, validate once the
      // store holds the registration; a model validates as an operation the strip shows, a dataset by preview ----
      const lifecycles = new Map();
      function lifecycle(job) {
        if (!lifecycles.has(job.id)) lifecycles.set(job.id, { host: el("span"), registered: null });
        const stage = lifecycles.get(job.id);
        const name = job.repository.split("/").pop();
        const cell = el("span", { class: "row" });
        if (job.state !== "succeeded") return cell;
        const report = (node) => { stage.host.replaceChildren(node); };
        const register = el("button", { class: "btn alt", text: stage.registered ? "registered" : "register", disabled: !!stage.registered, onclick: async () => {
          try {
            stage.registered = await overgo.api.post("/library/register", job.kind === "datasets"
              ? { kind: "dataset", name, directory: job.destination }
              : { kind: "model", path: job.destination, projector: job.projector || "" });
            report(el("span", { class: "note" }, "registered ", overgo.artifactLink(stage.registered.recipe || stage.registered.dataset),
              stage.registered.projector ? " with projector " + (stage.registered.media || []).join("/") : ""));
            jobPoller.start();
          } catch (err) { report(overgo.errorBanner(overgo.friendlyError(err))); }
        } });
        const validate = el("button", { class: "btn", text: "validate", disabled: !stage.registered, onclick: async () => {
          try {
            if (job.kind === "datasets") {
              const preview = await overgo.api.post("/datasets/preview", { name, position: 0, limit: 1 });
              report(el("span", { class: "note", text: "validated: " + (preview.rows || []).length + " row previewed" }));
            } else {
              const admitted = await overgo.api.post("/library/validate", { path: stage.registered.path, projector: stage.registered.projector || "" });
              report(el("span", { class: "note" }, "validating in operation ", overgo.artifactLink(admitted.operation), " · " + admitted.prompts + " prompts" + (admitted.projector ? " · projector" : "")));
              history.pushState({}, "", "?operation=" + encodeURIComponent(admitted.operation));
              window.dispatchEvent(new PopStateEvent("popstate"));
            }
          } catch (err) { report(overgo.errorBanner(overgo.friendlyError(err))); }
        } });
        cell.append(register, validate, stage.host);
        return cell;
      }
      const jobPoller = overgo.poller(async (signal) => {
        const listed = await overgo.api.get("/hub/downloads", { signal });
        jobsBody.replaceChildren(...(listed.downloads || []).map((job) => {
          const progress = job.total > 0 ? Math.floor(job.received * 100 / job.total) + "%" : "";
          const state = job.state === "failed" ? el("span", { class: "tag control", text: "failed" }) : job.state === "succeeded" ? el("span", { class: "tag user_defined", text: "done" }) : el("span", { class: "tag byte", text: progress || "running" });
          return overgo.tableRow([job.repository, state, job.file || (job.state === "failed" ? job.error : ""), job.total > 0 ? fmt.bytes(job.received) + " / " + fmt.bytes(job.total) : (job.files ? job.files + " files" : ""), lifecycle(job)]);
        }));
      }, 1000);

      panel.append(
        el("div", { class: "section-title", text: "Local models" }),
        catalogNote,
        el("table", { class: "grid" }, el("thead", null, overgo.headerRow(["model", "file", "capabilities", ""])), catalogBody),
        el("div", { class: "section-title", text: "Hugging Face" }),
        el("div", { class: "row" }, kind, query, searchButton, searchCancel, searchNote),
        el("table", { class: "grid" }, el("thead", null, overgo.headerRow(["repository", "downloads", "likes", ""])), resultsBody),
        el("div", { class: "section-title", text: "Downloads" }),
        jobsNote,
        el("table", { class: "grid" }, el("thead", null, overgo.headerRow(["repository", "state", "file", "progress", "register · validate"])), localBody, jobsBody));

      // ---- a local model: a GGUF (or its directory) already on disk, with its projector, rides the same lifecycle ----
      const localModel = el("input", { class: "text", placeholder: "model GGUF or directory on disk" });
      const localProjector = el("input", { class: "text", placeholder: "projector GGUF (optional)" });
      const localButton = el("button", { class: "btn alt", text: "register a local model", onclick: () => {
        const path = localModel.value.trim(); if (!path) return;
        const job = { id: "local:" + path, kind: "models", state: "succeeded", repository: path, destination: path, projector: localProjector.value.trim() };
        localBody.replaceChildren(overgo.tableRow([path, el("span", { class: "tag user_defined", text: "local" }), job.projector, "", lifecycle(job)]));
      } });
      jobsNote.after(el("div", { class: "row" }, localModel, localProjector, localButton));

      // ---- a hosted provider: the declaration the CLI file carries, committed through the library route; the catalog relists ----
      const providerFields = ["name", "endpoint", "key variable", "model ids (comma-separated)", "context length (optional)"].map((placeholder) => el("input", { class: "text", placeholder, "aria-label": "provider " + placeholder }));
      const providerNote = el("span", { class: "note" });
      const providerListing = el("span", { class: "row" });
      const providerValues = () => providerFields.map((field) => field.value.trim());
      const providerButton = el("button", { class: "btn alt", text: "declare a hosted provider", onclick: async () => {
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
      panel.append(el("div", { class: "section-title", text: "Hosted providers" }), el("div", { class: "row" }, ...providerFields, listButton, providerButton), providerListing, providerNote);

      refreshCatalog();
      this.onActivate = () => jobPoller.start();
      this.onDeactivate = () => jobPoller.stop();
      jobPoller.start();
    },
  });
})();
