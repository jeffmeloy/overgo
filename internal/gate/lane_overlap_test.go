package gate

import (
	"slices"
	"testing"

	"overgo/internal/automationcheck"
)

// TestDeferredLaneSerialWiring keeps replayed browser, model and CUDA checks
// in separate waves after a failed obligation. Ordinary gates retain their
// shared schedule, but replay must not repeat the VRAM and browser contention
// that made the obligation fail.
func TestDeferredLaneSerialWiring(t *testing.T) {
	t.Parallel()
	g := &gateContext{repo: t.TempDir(), paths: []string{"internal/server/webui_shell.go"}}
	byName := map[string]automationcheck.Descriptor{}
	var invocations []automationcheck.Invocation
	for _, check := range g.pipelineChecks() {
		byName[check.Descriptor.Name] = check.Descriptor
		invocations = append(invocations, automationcheck.Invocation{Check: check.Descriptor})
	}

	wired := map[string][]string{}
	for _, invocation := range wireLaneRunnerDependencies(invocations) {
		wired[invocation.Check.Name] = invocation.Check.Dependencies
	}
	if !slices.Contains(wired[testDeviceCheckName], testPlanCheckName) ||
		!slices.Contains(wired[testDeviceCheckName], automationcheck.ModelJourneyCheckName) ||
		!slices.Contains(wired[automationcheck.ModelJourneyCheckName], automationcheck.WebUICheckName) {
		t.Fatalf("deferred lane order lost: device=%v model=%v", wired[testDeviceCheckName], wired[automationcheck.ModelJourneyCheckName])
	}

	peers := []string{"device", automationcheck.WebUICheckName, automationcheck.ModelJourneyCheckName}
	for _, lane := range peers {
		descriptor, ok := byName[lane]
		if !ok {
			t.Fatalf("pipeline lacks the %s lane", lane)
		}
		for _, other := range peers {
			if other != lane && slices.Contains(descriptor.Dependencies, other) {
				t.Fatalf("lane %s depends on peer lane %s; they cannot co-schedule", lane, other)
			}
		}
		for _, resource := range descriptor.Resources {
			if resource.Exclusive {
				t.Fatalf("lane %s holds %s exclusively and cannot overlap shared correctness work", lane, resource.Name)
			}
		}
	}
}

// A merge or failed-lane replay pays for one reliable pass by ordering the
// browser and model journeys before the CUDA package batch. Normal gates keep
// their existing overlap; this checks the exceptional schedule itself.
func TestMergeReplaySerializesHeavyLanes(t *testing.T) {
	t.Parallel()
	checks := (&gateContext{repo: t.TempDir(), serializeLanes: true}).pipelineChecks()
	byName := map[string]automationcheck.Descriptor{}
	for _, check := range checks {
		byName[check.Descriptor.Name] = check.Descriptor
	}
	for name, prerequisite := range map[string]string{
		automationcheck.WebUICheckName:        testOwnersCheckName,
		automationcheck.ModelJourneyCheckName: automationcheck.WebUICheckName,
		testDeviceCheckName:                   automationcheck.ModelJourneyCheckName,
		"device":                              testDeviceCheckName,
	} {
		if !slices.Equal(byName[name].Dependencies, []string{prerequisite}) {
			t.Errorf("%s dependencies = %v; want %s", name, byName[name].Dependencies, prerequisite)
		}
	}
}
