package gate

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

// TestSharedLanesCoSchedule pins the lane overlap: the browser lane shares
// the device like the device lane. Their prerequisites remain explicit;
// eligible shared consumers overlap while exclusive measurements wait.
func TestSharedLanesCoSchedule(t *testing.T) {
	g := &gateContext{repo: t.TempDir(), paths: []string{"internal/server/webui_shell.go"}}
	byName := map[string]automationcheck.Descriptor{}
	for _, check := range g.pipelineChecks() {
		byName[check.Descriptor.Name] = check.Descriptor
	}
	for _, lane := range []string{"device", automationcheck.WebUICheckName} {
		descriptor, ok := byName[lane]
		if !ok {
			t.Fatalf("pipeline lacks the %s lane", lane)
		}
		prerequisite := "test-device"
		if lane == automationcheck.WebUICheckName {
			prerequisite = "test-owners"
		}
		if !slices.Equal(descriptor.Dependencies, []string{prerequisite}) {
			t.Fatalf("%s depends on %v, want %s", lane, descriptor.Dependencies, prerequisite)
		}
		for _, resource := range descriptor.Resources {
			if resource.Exclusive {
				t.Fatalf("%s holds %s exclusively", lane, resource.Name)
			}
		}
	}
	if !slices.Equal(byName["commit"].Dependencies, []string{"test", "device", automationcheck.WebUICheckName}) {
		t.Fatalf("commit depends on %v", byName["commit"].Dependencies)
	}

	shared := []automationcheck.Resource{{Name: "device"}}
	exclusive := []automationcheck.Resource{{Name: "device", Exclusive: true}}
	invocation := func(name string, resources []automationcheck.Resource, dependencies ...string) automationcheck.Invocation {
		return automationcheck.Invocation{Check: automationcheck.Descriptor{
			Name: name, Phase: runrecord.PhaseTest, Always: true, Resources: resources, Dependencies: dependencies,
		}}
	}
	invocations := []automationcheck.Invocation{
		invocation("build", nil), invocation("vet", nil),
		invocation("test", nil, "vet", "build"),
		invocation("device", shared, "test"),
		invocation(automationcheck.WebUICheckName, shared, "test"),
		invocation("measure", exclusive, "test"),
		invocation("commit", nil, "test", "device", automationcheck.WebUICheckName, "measure"),
	}
	var mutex sync.Mutex
	active := map[string]bool{}
	overlaps := map[string][]string{}
	execute := func(ctx context.Context, current automationcheck.Invocation) (automationcheck.Evidence, error) {
		mutex.Lock()
		active[current.Check.Name] = true
		for name := range active {
			if name != current.Check.Name {
				overlaps[current.Check.Name] = append(overlaps[current.Check.Name], name)
				overlaps[name] = append(overlaps[name], current.Check.Name)
			}
		}
		mutex.Unlock()
		time.Sleep(80 * time.Millisecond)
		mutex.Lock()
		delete(active, current.Check.Name)
		mutex.Unlock()
		return automationcheck.Evidence{}, nil
	}
	results, err := automationcheck.ExecuteDAG(t.Context(), invocations, nil, execute)
	if err != nil || len(results) != len(invocations) {
		t.Fatalf("DAG execution = %d results, %v", len(results), err)
	}
	for _, pair := range [][2]string{{"device", automationcheck.WebUICheckName}} {
		if !slices.Contains(overlaps[pair[0]], pair[1]) {
			t.Fatalf("%s and %s did not run in one wave: overlaps=%v", pair[0], pair[1], overlaps)
		}
	}
	for _, other := range overlaps["measure"] {
		if other == "device" || other == automationcheck.WebUICheckName {
			t.Fatalf("the exclusive measurement overlapped %s: %v", other, overlaps["measure"])
		}
	}
}
