package gate

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync/atomic"
	"testing"

	"overgo/internal/automationcheck"
)

// A channel handshake proves overlap without relying on sleep durations.
func TestSharedBrowserDependencyOverlap(t *testing.T) {
	t.Parallel()
	checks := (&gateContext{repo: t.TempDir()}).pipelineChecks()
	for _, check := range checks {
		if check.Descriptor.Name == automationcheck.WebUICheckName && !slices.Equal(check.Descriptor.Dependencies, []string{"test-owners"}) {
			t.Fatalf("browser unnecessarily serialized: %v", check.Descriptor.Dependencies)
		}
	}
	ctx := t.Context()
	deviceStarted, browserStarted := make(chan struct{}), make(chan struct{})
	var ownersDone, deviceDone, browserDone atomic.Bool
	for i := range checks {
		checks[i].Run = func(ctx context.Context, invocation automationcheck.Invocation) (bool, string, error) {
			switch invocation.Check.Name {
			case "test-owners":
				ownersDone.Store(true)
			case "test-device", automationcheck.WebUICheckName:
				if !ownersDone.Load() {
					return false, "", fmt.Errorf("%s preceded owner acceptance", invocation.Check.Name)
				}
				var peer <-chan struct{}
				if invocation.Check.Name == "test-device" {
					close(deviceStarted)
					peer = browserStarted
				} else {
					close(browserStarted)
					peer = deviceStarted
				}
				select {
				case <-peer:
				case <-ctx.Done():
					return false, "", ctx.Err()
				}
				if invocation.Check.Name == "test-device" {
					deviceDone.Store(true)
				} else {
					browserDone.Store(true)
				}
			case "commit":
				if !deviceDone.Load() || !browserDone.Load() {
					return false, "", errors.New("commit preceded required checks")
				}
			}
			return false, "", nil
		}
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	results, err := automationcheck.ExecuteDAG(ctx, invocations, nil, automationcheck.Run)
	if err != nil || len(results) != len(invocations) {
		t.Fatalf("DAG results=%d/%d err=%v", len(results), len(invocations), err)
	}
	for _, result := range results {
		if result.Err != nil {
			t.Fatalf("%s: %v", result.Invocation.Check.Name, result.Err)
		}
	}
	if !deviceDone.Load() || !browserDone.Load() {
		t.Fatal("required consumers did not complete")
	}
	t.Log("actual pipeline overlaps browser and shared dependent tests after owners; commit joins both; no checks removed")
}
