package gate

import (
	"slices"
	"testing"

	"overgo/internal/automationcheck"
)

// TestDeferredLaneOverlapWiring holds the deferred runner's schedule against the
// same frozen candidate: the runner wires the device test group to follow the
// test plan it consumes, the peer lanes -- device, browser and model journeys --
// depend on none of one another so they co-schedule in one wave, and each holds
// its shared resource non-exclusively, so ordinary correctness lanes overlap
// while a declared exclusive measurement still serializes.
func TestDeferredLaneOverlapWiring(t *testing.T) {
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
	if !slices.Contains(wired[testDeviceCheckName], testPlanCheckName) {
		t.Fatalf("the device test group is not wired to the test plan it consumes: %v", wired[testDeviceCheckName])
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
