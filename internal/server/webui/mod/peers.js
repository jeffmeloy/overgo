(function () {
  "use strict";

  window.overgo.registerTab({
    id: "peers",
    async mount(panel, overgo) {
      const { api, el, fmt } = overgo;
      const status = el("div", { class: "note" });
      const inventoryHost = el("div");
      const detailHost = el("div");
      const enrollmentHost = el("div");
      const placementHost = el("div");
      const evidenceHost = el("div");
      panel.append(
        el("div", { class: "section-title", text: "Peer control plane" }), status,
        el("div", { class: "section-title", text: "Inventory, peer labels, and capacity" }), inventoryHost,
        el("div", { class: "section-title", text: "Peer detail" }), detailHost,
        el("div", { class: "section-title", text: "Enrollment" }), enrollmentHost,
        el("div", { class: "section-title", text: "Placement rules" }), placementHost,
        el("div", { class: "section-title", text: "Staging, attempts, and logs" }), evidenceHost);

      const [enrollmentSchema, placementSchema] = await Promise.all([
        api.get("/workspace/schema?id=peer-enrollment"),
        api.get("/workspace/schema?id=peer-placement"),
      ]);
      const enrollmentForm = overgo.schemaForm(enrollmentSchema, {});
      const placementForm = overgo.schemaForm(placementSchema, {
        task: "generation", minimum_replicas: 1, maximum_replicas: 1,
        target_concurrency: 1, concurrency_per_replica: 1,
        allow_local: true, allow_peers: true, maximum_attempts: 2,
      });
      const enroll = el("button", { class: "btn", text: "Enroll approved peer" });
      const compile = el("button", { class: "btn", text: "Compile placement" });
      const reconcile = el("button", { class: "btn alt", text: "Reconcile placement", disabled: true });
      enrollmentHost.append(enrollmentForm.element, enroll);
      placementHost.append(placementForm.element, compile, reconcile);

      let inventory = { peers: [], truncated: false };
      let selectedPeer = "";
      let compiledPlan = null;

      function openGlobalOperation(id) {
        const url = new URL(window.location.href);
        url.searchParams.set("operation", id);
        window.history.pushState(null, "", url);
        window.dispatchEvent(new PopStateEvent("popstate"));
      }

      const artifactLink = overgo.artifactLink;

      function renderDetail() {
        const item = (inventory.peers || []).find((peer) => peer.peer === selectedPeer);
        if (!item) {
          detailHost.replaceChildren(el("div", { class: "note", text: "Select a peer to inspect exact authority." }));
          return;
        }
        const state = item.state || {};
        const capacity = item.capacity || {};
        const actions = el("div", { class: "row" }, ...[["active", "Drain", "draining"], ["draining", "Retire after drain", "retired"]]
          .filter(([from]) => state.state === from).map(([, text, to]) => el("button", { class: "btn alt", text, onclick: () => transition(item.peer, to) })));
        detailHost.replaceChildren(el("div", { class: "card" },
          el("strong", { text: item.enrollment.name }), " / ", el("span", { text: state.state || "unknown" }),
          el("div", { class: "mono", text: item.peer }),
          el("div", { text: "environment / " + capacity.environment }),
          el("div", { text: "tasks / " + (capacity.tasks || []).join(", ") }),
          el("div", { text: "lease / " + (capacity.available ? "available" : "unavailable") +
            (capacity.lease_expires_unix_ns ? " until " + capacity.lease_expires_unix_ns : "") }),
          item.refusal ? overgo.errorBanner(item.refusal) : el("span", { class: "tag user_defined", text: "eligible" }),
          actions));
      }

      function renderInventory() {
        const table = el("table", { class: "grid" }, overgo.headerRow(["label", "state", "capacity", "lease", "authority"]));
        for (const item of inventory.peers || []) {
          const capacity = item.capacity || {};
          table.appendChild(overgo.tableRow([el("button", { class: "link-button", text: item.enrollment.name }),
            (item.state && item.state.state) || "unknown", (capacity.tasks || []).join(", ") || "none",
            capacity.available ? "available" : "unavailable", item.refusal || "eligible"],
            { onclick: () => { selectedPeer = item.peer; renderDetail(); } }));
        }
        if (inventory.truncated) table.appendChild(el("caption", { text: "Projection truncated by server policy" }));
        inventoryHost.replaceChildren(table);
        renderDetail();
      }

      async function refreshInventory() { inventory = await api.get("/peers"); renderInventory(); }

      async function transition(peer, state) {
        try {
          await api.post("/peers/state", { peer, state, changed_unix_ns: 0 });
          status.textContent = state + " / " + fmt.shortID(peer);
          await refreshInventory();
        } catch (err) { status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      }

      enroll.addEventListener("click", async () => {
        if (!enrollmentForm.validate()) { status.textContent = "Complete every enrollment field"; return; }
        try {
          const result = await api.post("/peers/enroll", enrollmentForm.value());
          enrollmentForm.markSaved();
          status.textContent = "enrolled / " + fmt.shortID(result.peer);
          selectedPeer = result.peer;
          await refreshInventory();
        } catch (err) { status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      });

      function placementRequest(value) {
        const policyFields = ["minimum_replicas", "maximum_replicas", "target_concurrency", "concurrency_per_replica",
          "maximum_measured_ns", "maximum_device_bytes", "allow_local", "allow_peers"];
        return {
          model: value.model, task: value.task, now_unix_ns: 0, local_observation: value.local_observation,
          policy: Object.fromEntries(policyFields.map((field) => [field, value[field]])),
          peers: (value.peer_candidates || []).map((line) => { const parts = line.split(/\s+/); return { peer: parts[0], compatibility: parts[1] }; }),
        };
      }

      compile.addEventListener("click", async () => {
        if (!placementForm.validate()) { status.textContent = "Complete every applicable placement field"; return; }
        try {
          compiledPlan = await api.post("/peers/placement", placementRequest(placementForm.value()));
          placementForm.markSaved();
          reconcile.disabled = false;
          const replicaCards = (compiledPlan.replicas || []).map((replica) => el("div", { class: "card" },
            el("strong", { text: "replica " + replica.index }),
            el("div", { text: replica.peer ? "peer / " + replica.peer : "local" }),
            el("div", { text: "latency / " + replica.measured_ns + " ns" }),
            ...((replica.locality || []).map((location) => el("div", {}, "artifact / ", artifactLink(location.artifact))))));
          for (const refusal of compiledPlan.refusals || []) replicaCards.push(overgo.errorBanner(refusal.reason + " / " + refusal.detail));
          evidenceHost.replaceChildren(...replicaCards);
          status.textContent = "placement compiled / " + fmt.shortID(compiledPlan.identity);
        } catch (err) { compiledPlan = null; reconcile.disabled = true; status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      });

      reconcile.addEventListener("click", async () => {
        if (!compiledPlan) return;
        try {
          const value = placementForm.value();
          const result = await api.post("/peers/reconcile", {
            plan: compiledPlan, previous: [], policy: { maximum_attempts: value.maximum_attempts },
            changed_unix_ns: 0,
          });
          status.textContent = "reconciliation admitted / " + fmt.shortID(result.operation);
          openGlobalOperation(result.operation);
        } catch (err) { status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err))); }
      });

      async function renderEvidence(operation) {
        const evidence = await api.get("/peers/evidence?operation=" + encodeURIComponent(operation));
        if (!(evidence.attempts || []).length) return;
        evidenceHost.replaceChildren(overgo.table(["phase", "outcome", "attempt", "failure / log", "artifacts"], (evidence.attempts || []).map(({ value: attempt }) => [
          attempt.phase, attempt.outcome, String(attempt.attempt), attempt.failure || "completed", el("span", {}, ...((attempt.artifacts || []).map(artifactLink)))])));
      }

      // An evidence read that fails says so where the evidence would stand.
      const showEvidence = (data) => { if (data.status && data.status.id) renderEvidence(data.status.id).catch((err) => evidenceHost.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)))); };
      await refreshInventory();
      const stream = new AbortController();
      api.events("/peers/stream", (event, data) => {
        if (event === "peer.inventory") { inventory = data; renderInventory(); }
        if (event === "operation") showEvidence(data);
      }, { signal: stream.signal }).catch((err) => {
        if (err.name !== "AbortError") status.replaceChildren(overgo.errorBanner(overgo.friendlyError(err)));
      });
      overgo.runtimeEvents.subscribe((event, data) => { if (event === "operation") showEvidence(data); });
      return () => { stream.abort(); enrollmentForm.dispose(); placementForm.dispose(); };
    },
  });
})();
