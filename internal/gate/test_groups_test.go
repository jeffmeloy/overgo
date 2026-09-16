package gate

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/processlock"
)

// Owners follow required preparation; lanes overlap after owner acceptance.
// Commit joins the required work. Missing preparation fails explicitly;
// store contention waits for the writer's release.
func TestTestGroupsOverlapUnderLedger(t *testing.T) {
	t.Parallel()
	g := &gateContext{repo: t.TempDir(), paths: []string{"internal/gate/gate.go"}}
	checks := g.pipelineChecks()
	byName := map[string]automationcheck.Descriptor{}
	for _, check := range checks {
		byName[check.Descriptor.Name] = check.Descriptor
	}
	for name, prerequisites := range map[string][]string{
		testPlanCheckName: {"acceptance"}, testOwnersCheckName: {testPlanCheckName},
		testDeviceCheckName: {testOwnersCheckName}, testRestCheckName: {testDeviceCheckName},
	} {
		if !slices.Equal(byName[name].Dependencies, prerequisites) {
			t.Fatalf("%s depends on %v, want %v", name, byName[name].Dependencies, prerequisites)
		}
	}
	for _, lane := range []string{"device", automationcheck.WebUICheckName, automationcheck.ModelJourneyCheckName} {
		prerequisite := "test-device"
		if lane != "device" {
			prerequisite = "test-owners"
		}
		if !slices.Equal(byName[lane].Dependencies, []string{prerequisite}) {
			t.Fatalf("%s depends on %v, want the device check of the remaining groups alone", lane, byName[lane].Dependencies)
		}
	}
	if !slices.Equal(byName["commit"].Dependencies, []string{"test", "device", automationcheck.WebUICheckName, automationcheck.ModelJourneyCheckName}) {
		t.Fatalf("commit depends on %v", byName["commit"].Dependencies)
	}
	if phaseReusesEvidence(testOwnersCheckName) || phaseReusesEvidence(testDeviceCheckName) || !phaseOwnsPath(testOwnersCheckName, "internal/gate/gate.go") {
		t.Fatal("package checks must retain source binding and reuse individual receipts")
	}

	// The remaining groups run beside the lanes: each pair proves its wave
	// by meeting, and the spans keep the order between phases.
	type span struct{ start, end time.Time }
	var mutex sync.Mutex
	spans := map[string]span{}
	restBesideDevice := newRendezvous("test", "device")
	deviceBesideBrowser := newRendezvous(testDeviceCheckName, automationcheck.WebUICheckName)
	for index := range checks {
		checks[index].Run = func(_ context.Context, invocation automationcheck.Invocation) (bool, string, error) {
			started := time.Now()
			restBesideDevice.meet(invocation.Check.Name)
			deviceBesideBrowser.meet(invocation.Check.Name)
			mutex.Lock()
			spans[invocation.Check.Name] = span{started, time.Now()}
			mutex.Unlock()
			return false, "", nil
		}
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := automationcheck.ExecuteDAG(t.Context(), invocations, nil, automationcheck.Run); err != nil {
		t.Fatal(err)
	}
	for _, follower := range []string{"test", "device", automationcheck.WebUICheckName, automationcheck.ModelJourneyCheckName} {
		prerequisite := "test-device"
		if follower == automationcheck.WebUICheckName || follower == automationcheck.ModelJourneyCheckName {
			prerequisite = "test-owners"
		}
		if spans[follower].start.Before(spans[prerequisite].end) {
			t.Fatalf("%s started before %s passed", follower, prerequisite)
		}
		if spans["commit"].start.Before(spans[follower].end) {
			t.Fatalf("commit started before %s ended", follower)
		}
	}
	if skipped, err := (&gateContext{}).stepTestRest(t.Context()); err == nil || skipped {
		t.Fatalf("rest without prepared groups = skipped %t, %v; want failure", skipped, err)
	}
	if skipped, err := (&gateContext{testPlan: &testGroups{}}).stepTestRest(t.Context()); err != nil || !skipped {
		t.Fatalf("rest with an explicitly empty scope = skipped %t, %v; want inapplicable", skipped, err)
	}

	// The gate's store handle waits for another writer holding the store
	// lock, as a lane recording receipts beside the test groups does.
	store := &gateContext{repo: t.TempDir(), storePath: gateStorePath}
	environment, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("environment"))
	if err != nil {
		t.Fatal(err)
	}
	store.environment.ID = environment
	if err := os.MkdirAll(filepath.Join(store.repo, gateStorePath), 0o700); err != nil {
		t.Fatal(err)
	}
	// The writer releases beside the admission; a handle that refused the
	// held lock instead of waiting would answer with the contention error,
	// and the wait itself is the store's contract, held by its own tests.
	held, err := processlock.Acquire(filepath.Join(store.repo, gateStorePath, "overgodb.lock"), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = held.Close() }()
	opened, err := store.openStore()
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
}
