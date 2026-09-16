package gate

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestDeferredLaneObligations holds the mechanism that lets a commit land
// before its lanes: the rewired graph makes the commit follow the host tests
// alone and lists the three lanes as deferred; a DAG run with refusing lane
// stubs executes the commit and never a lane; the obligation chain admits,
// refuses with a live or a dead runner, forces the lanes inline on failure
// and resolves on supersession; and a successful record carries the pending
// obligation and the supersession under the store's alias.
func TestDeferredLaneObligations(t *testing.T) {
	t.Parallel()
	dependencies := func(definitions []automationcheck.Check, name string) []string {
		for _, definition := range definitions {
			if definition.Descriptor.Name == name {
				return definition.Descriptor.Dependencies
			}
		}
		t.Fatalf("definition %s absent", name)
		return nil
	}
	compiler := scopeCompilerFixture(t)
	declared := compiler.pipelineChecks()
	if !slices.Contains(dependencies(declared, "commit"), "device") || !slices.Contains(dependencies(declared, "commit"), automationcheck.WebUICheckName) {
		t.Fatalf("declared commit dependencies = %v", dependencies(declared, "commit"))
	}
	if compiler.laneDeferral() != nil {
		t.Fatal("a gate that keeps its lanes listed deferrals")
	}
	compiler.deferLanes = true
	rewired := compiler.rewireDeferredLanes(compiler.pipelineChecks())
	if !slices.Equal(dependencies(rewired, "commit"), []string{testRestCheckName}) ||
		!slices.Equal(dependencies(rewired, testRestCheckName), []string{testOwnersCheckName}) {
		t.Fatalf("rewired dependencies: commit=%v test=%v", dependencies(rewired, "commit"), dependencies(rewired, testRestCheckName))
	}
	if deferred := compiler.laneDeferral(); len(deferred) != 4 || !deferred[testDeviceCheckName] || !deferred["device"] || !deferred[automationcheck.WebUICheckName] || !deferred[automationcheck.ModelJourneyCheckName] {
		t.Fatalf("lane deferral = %v", deferred)
	}

	g, _, tree := verificationBatchFixture(t, "pass")
	g.deferLanes = true
	refuse := func() (bool, error) { return false, errors.New("a deferred lane ran before the commit") }
	pass := func() (bool, error) { return false, nil }
	checks := []automationcheck.Check{
		gateCheck(testOwnersCheckName, runrecord.PhaseTest, pass),
		gateCheck(testDeviceCheckName, runrecord.PhaseTest, refuse),
		gateCheck(testRestCheckName, runrecord.PhaseTest, pass),
		gateCheck("device", runrecord.PhaseTest, refuse),
		gateCheck(automationcheck.WebUICheckName, runrecord.PhaseTest, refuse),
		gateCheck(automationcheck.ModelJourneyCheckName, runrecord.PhaseTest, refuse),
		gateCheck("commit", runrecord.PhasePackage, pass),
	}
	graph := map[string][]string{
		testDeviceCheckName: {testOwnersCheckName}, testRestCheckName: {testDeviceCheckName}, "device": {testDeviceCheckName},
		automationcheck.WebUICheckName: {testOwnersCheckName}, automationcheck.ModelJourneyCheckName: {testOwnersCheckName}, "commit": {testRestCheckName, "device", automationcheck.WebUICheckName, automationcheck.ModelJourneyCheckName},
	}
	for index := range checks {
		checks[index].Descriptor.Dependencies = graph[checks[index].Descriptor.Name]
	}
	checks = g.rewireDeferredLanes(checks)
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := automationcheck.BindManifestPlan(
		testutil.ArtifactID(t, artifact.KindProfile, "deferred lanes base"),
		testutil.ArtifactID(t, artifact.KindProfile, "deferred lanes candidate"),
		strings.Repeat("a", 64), candidateTreeKey(tree),
		automationcheck.Surface{Identity: "deferred lanes fixture"}, automationcheck.Impact{}, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	g.manifestPlan = &manifest
	for index, invocation := range invocations {
		if invocations[index], err = g.bindCheckExecution(manifest, invocation, manifest.CandidateManifest); err != nil {
			t.Fatal(err)
		}
	}
	cache := g.loadRetryCache()
	results, err := g.executeChecks(invocations, nil, map[artifact.ID]artifact.ID{}, &cache, nil, g.laneDeferral())
	if err != nil {
		t.Fatal(err)
	}
	var executed []string
	for _, result := range results {
		if !result.Invocation.ID.Valid() {
			continue
		}
		if result.Err != nil {
			t.Fatalf("%s failed: %v", result.Invocation.Check.Name, result.Err)
		}
		executed = append(executed, result.Invocation.Check.Name)
	}
	slices.Sort(executed)
	if !slices.Equal(executed, []string{"commit", testRestCheckName, testOwnersCheckName}) {
		t.Fatalf("executed = %v, want the commit after the host tests alone", executed)
	}
	// The commit's admission requires terminal evidence for every planned
	// check except the lanes the gate recorded as deferred.
	terminal := map[string]automationcheck.Evidence{}
	for _, result := range results {
		if result.Invocation.ID.Valid() && result.Invocation.Check.Name != "commit" {
			terminal[result.Invocation.Check.Name] = result.Evidence
		}
	}
	if err := validateManifestCommitAdmission(manifest, terminal, g.laneDeferral()); err != nil {
		t.Fatalf("commit admission refused the deferred lanes: %v", err)
	}
	if err := validateManifestCommitAdmission(manifest, terminal, nil); err == nil || !strings.Contains(err.Error(), "lacks terminal evidence") {
		t.Fatalf("commit admission without deferral = %v", err)
	}
	if err := g.closeStore(); err != nil {
		t.Fatal(err)
	}
	// The runner satisfies the owners' tests without executing them, so the
	// device group must follow the test plan directly; the first live run
	// started test-device beside test-plan and found no scope.
	wired := wireLaneRunnerDependencies(slices.Clone(invocations))
	for _, invocation := range wired {
		if invocation.Check.Name == testDeviceCheckName && !slices.Contains(invocation.Check.Dependencies, testPlanCheckName) {
			t.Fatalf("runner left test-device without the test plan: %v", invocation.Check.Dependencies)
		}
	}

	repo := t.TempDir()
	if err := os.Mkdir(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if debt, err := requireLaneObligationsResolved(repo, store); debt != nil || err != nil {
		t.Fatalf("empty store: debt=%v err=%v", debt, err)
	}
	commit := strings.Repeat("c", 40)
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "deferred lanes preparation")
	result := testutil.ArtifactID(t, artifact.KindEvidence, "deferred lanes result")
	laneRun := testutil.ArtifactID(t, artifact.KindEvidence, "deferred lanes run")
	now := time.Now()
	pending, err := runrecord.NewGateLaneObligation(commit, preparation, result, []string{automationcheck.WebUICheckName}, []string{"docs/plan.json"}, now)
	if err != nil {
		t.Fatal(err)
	}
	publish := func(state runrecord.GateLaneObligation, previous *artifact.ID) {
		t.Helper()
		batch, err := state.Batch(previous)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
	}
	publish(pending, nil)
	if _, err := requireLaneObligationsResolved(repo, store); err == nil || !strings.Contains(err.Error(), "-lanes") {
		t.Fatalf("pending obligation without a runner: %v", err)
	}
	locator := gateLanesLocator{Version: artifact.InitialDocumentVersion, State: runrecord.LaneObligationRunning, Obligation: pending.ID, CodeCommit: commit, PID: os.Getpid(), Updated: now}
	if err := writeJSON(repo, gateLanesLocatorFile, locator, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := requireLaneObligationsResolved(repo, store); err == nil || !strings.Contains(err.Error(), "still running") {
		t.Fatalf("pending obligation with a live runner: %v", err)
	}
	running, err := pending.Transition(runrecord.LaneObligationRunning, artifact.ID{}, now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	publish(running, &pending.ID)
	resumed, err := (&gateContext{repo: repo, store: store}).resumeLaneObligation(running)
	if err != nil || resumed.ID != running.ID {
		t.Fatalf("interrupted runner did not retain its obligation: %+v %v", resumed, err)
	}
	retained, found, err := runrecord.CurrentGateLaneObligation(t.Context(), store)
	if err != nil || !found || retained.ID != running.ID {
		t.Fatalf("resuming changed the durable obligation: %+v %v", retained, err)
	}
	failed, err := running.Transition(runrecord.LaneObligationFailed, laneRun, now.Add(2*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	publish(failed, &running.ID)
	debt, err := requireLaneObligationsResolved(repo, store)
	if err != nil || debt == nil || debt.ID != failed.ID {
		t.Fatalf("failed obligation: debt=%v err=%v", debt, err)
	}

	lifecycle, err := runrecord.NewGatePreparation(strings.Repeat("b", 64), testutil.ArtifactID(t, artifact.KindEvidence, "deferred lanes environment"), now)
	if err != nil {
		t.Fatal(err)
	}
	inline := &gateContext{repo: repo, store: store, paths: []string{"docs/plan.json"}, preparation: lifecycle, laneDebt: debt}
	batch := artifact.Batch{Key: "test/deferred-lanes/supersede"}
	if err := inline.appendLaneObligation(&batch, commit, result); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	current, found, err := runrecord.CurrentGateLaneObligation(t.Context(), store)
	if err != nil || !found || current.State != runrecord.LaneObligationSuperseded || current.Outcome != result {
		t.Fatalf("after the inline lanes passed: %+v found=%v %v", current, found, err)
	}
	if debt, err := requireLaneObligationsResolved(repo, store); debt != nil || err != nil {
		t.Fatalf("superseded obligation still blocks: debt=%v err=%v", debt, err)
	}
	deferring := &gateContext{repo: repo, store: store, paths: []string{"docs/plan.json", "internal/gate/run.go"}, preparation: lifecycle, deferredLanes: []string{"device", testDeviceCheckName}}
	batch = artifact.Batch{Key: "test/deferred-lanes/pending"}
	if err := deferring.appendLaneObligation(&batch, commit, laneRun); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	current, _, err = runrecord.CurrentGateLaneObligation(t.Context(), store)
	if err != nil || current.State != runrecord.LaneObligationPending || !slices.Equal(current.Checks, []string{"device", testDeviceCheckName}) || deferring.laneObligation == nil || deferring.laneObligation.ID != current.ID {
		t.Fatalf("recorded obligation: %+v %v", current, err)
	}
	if _, err := requireLaneObligationsResolved(repo, store); err == nil || !strings.Contains(err.Error(), "-lanes") {
		t.Fatalf("new pending obligation without its runner: %v", err)
	}
}
