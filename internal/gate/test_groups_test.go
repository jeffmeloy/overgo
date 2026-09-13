package gate

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
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
	for _, lane := range []string{"device", automationcheck.WebUICheckName} {
		prerequisite := "test-device"
		if lane == automationcheck.WebUICheckName {
			prerequisite = "test-owners"
		}
		if !slices.Equal(byName[lane].Dependencies, []string{prerequisite}) {
			t.Fatalf("%s depends on %v, want the device check of the remaining groups alone", lane, byName[lane].Dependencies)
		}
	}
	if !slices.Equal(byName["commit"].Dependencies, []string{"test", "device", automationcheck.WebUICheckName}) {
		t.Fatalf("commit depends on %v", byName["commit"].Dependencies)
	}
	if phaseReusesEvidence(testOwnersCheckName) || phaseReusesEvidence(testDeviceCheckName) || !phaseOwnsPath(testOwnersCheckName, "internal/gate/gate.go") {
		t.Fatal("package checks must retain source binding and reuse individual receipts")
	}

	type span struct{ start, end time.Time }
	var mutex sync.Mutex
	spans := map[string]span{}
	for index := range checks {
		checks[index].Run = func(_ context.Context, invocation automationcheck.Invocation) (bool, string, error) {
			started := time.Now()
			time.Sleep(60 * time.Millisecond)
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
	overlaps := func(left, right string) bool {
		return spans[left].start.Before(spans[right].end) && spans[right].start.Before(spans[left].end)
	}
	for _, follower := range []string{"test", "device", automationcheck.WebUICheckName} {
		prerequisite := "test-device"
		if follower == automationcheck.WebUICheckName {
			prerequisite = "test-owners"
		}
		if spans[follower].start.Before(spans[prerequisite].end) {
			t.Fatalf("%s started before %s passed", follower, prerequisite)
		}
		if spans["commit"].start.Before(spans[follower].end) {
			t.Fatalf("commit started before %s ended", follower)
		}
	}
	if !overlaps("test", "device") || !overlaps("test-device", automationcheck.WebUICheckName) {
		t.Fatalf("the remaining groups did not run beside the lanes: %v", spans)
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
	held, err := processlock.Acquire(filepath.Join(store.repo, gateStorePath, "overgodb.lock"), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		time.Sleep(300 * time.Millisecond)
		_ = held.Close()
	}()
	began := time.Now()
	opened, err := store.openStore()
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if waited := time.Since(began); waited < 250*time.Millisecond {
		t.Fatalf("store admission did not wait for the writer: %s", waited)
	}
	if !slices.ContainsFunc(store.audit, func(line string) bool { return strings.HasPrefix(line, "store admission waited") }) && time.Since(began) > time.Second {
		t.Fatalf("a wait over a second went unaudited: %q", store.audit)
	}
}
