package main

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codeprofile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

func TestGateAdvisoryFindingPublication(t *testing.T) {
	store := mustGateValue(overgodb.Open(filepath.Join(t.TempDir(), "store")))
	defer store.Close()
	batch := artifact.Batch{Key: "gate/fixture"}
	honesty := []string{"magic backlog: 2 inherited uncatalogued constants"}
	if err := appendGateAdvisoryFinding(context.Background(), store, &batch, []string{"internal/p"}, honesty); err != nil {
		t.Fatal(err)
	}
	mustGateValue(store.Commit(context.Background(), batch))
	next := artifact.Batch{Key: "gate/fixture/repeat"}
	if err := appendGateAdvisoryFinding(context.Background(), store, &next, []string{"internal/p"}, honesty); err != nil ||
		len(next.Aliases) != 1 || next.Aliases[0].Previous == nil || *next.Aliases[0].Previous != batch.Aliases[0].Target {
		t.Fatalf("deduplicated finding = %+v, %v", next.Aliases, err)
	}
}

func TestGateSummarySeparatesBlockersAndAdvisories(t *testing.T) {
	gate := gateContext{
		start: time.Now(),
		steps: []runrecord.GateStep{
			{Name: "build", Outcome: runrecord.StepSucceeded},
			{Name: "test", Outcome: runrecord.StepReused},
			{Name: "claims", Outcome: runrecord.StepSkipped},
		},
		honesty: []string{
			"code profile: production=700 files/1000000 nodes and a large routine baseline",
			"code profile delta vs HEAD: production=+0 files/-20 nodes duplicate_excess=-12",
			"consumer census commit context=windows/amd64 delta: production=+1 test_only=+0 boundary=+0 zero=+0",
			"test scope: 2 direct + 1 dependent packages (derived from import graph)",
			"claims skipped: no changed path appears in compatibility.json",
			"magic backlog: 3 inherited uncatalogued constants",
		},
	}
	var output bytes.Buffer
	fmt.Fprintf(&output, gateProgressLine, "test", runrecord.HeartbeatRunning)
	gate.printSummary(&output, runrecord.OutcomeFailed, "test: exit status 1")
	text := output.String()
	if len(text) > 1000 || !strings.Contains(text, "phase=test heartbeat=running") ||
		!strings.Contains(text, "GATE FAILED") || !strings.Contains(text, "blocker: test: exit status 1") ||
		!strings.Contains(text, "ran=build | reused=test | skipped=claims | inapplicable=") ||
		!strings.Contains(text, "advisory: delta:") || !strings.Contains(text, "advisory: warning:") {
		t.Fatalf("gate output is not compact and decision-complete (%d bytes):\n%s", len(text), text)
	}
	if strings.Contains(text, "routine baseline") || strings.Contains(text, "claims skipped:") {
		t.Fatalf("gate output repeated stored routine evidence:\n%s", text)
	}
	profile := codeprofile.Profile{Clones: []codeprofile.Clone{{
		Nodes: 8, Functions: []string{"cmd/first/main.go:main", "cmd/second/main.go:main"},
	}}}
	if focus := profileReviewFocus(profile, []string{"cmd/first/main.go"}); !strings.Contains(focus, "exact_clone=none") {
		t.Fatalf("CLI wrapper clone reached agent output: %s", focus)
	}
}

func TestGateRunsAcceptanceBeforeExpensivePhases(t *testing.T) {
	steps := (&gateContext{}).pipelineChecks()
	positions := make(map[string]int, len(steps))
	for index, step := range steps {
		positions[step.Descriptor.Name] = index
	}
	for _, expensive := range []string{"vet", "build", "test", "device"} {
		if positions["acceptance"] >= positions[expensive] {
			t.Fatalf("acceptance position %d is not before %s at %d", positions["acceptance"], expensive, positions[expensive])
		}
	}
}

func TestMergeGateRunsAuthorityPreflightBeforeBroadTests(t *testing.T) {
	steps := (&gateContext{}).pipelineChecks()
	positions := make(map[string]int, len(steps))
	for index, step := range steps {
		positions[step.Descriptor.Name] = index
	}
	for _, authority := range []string{"fmt", "style", "manifest", "sbom", "claims", "docs", "magics"} {
		for _, expensive := range []string{"acceptance", "vet", "build", "test", "device"} {
			if positions[authority] >= positions[expensive] {
				t.Fatalf("authority step %s at %d follows %s at %d", authority, positions[authority], expensive, positions[expensive])
			}
		}
	}
}

func TestModularPipelineDeclaresApplicabilityAndResources(t *testing.T) {
	checks := (&gateContext{repo: t.TempDir(), paths: []string{"internal/cuda/kernel/load.go"}}).pipelineChecks()
	byName := make(map[string]automationcheck.Descriptor, len(checks))
	for _, check := range checks {
		byName[check.Descriptor.Name] = check.Descriptor
	}
	for _, name := range []string{"manifest", "sbom", "claims", "device"} {
		if len(byName[name].Triggers) == 0 || byName[name].Inapplicable == "" {
			t.Errorf("%s lacks modular applicability: %+v", name, byName[name])
		}
	}
	resources := byName["device"].Resources
	if len(resources) != 1 || resources[0].Name != "device" || !resources[0].Exclusive {
		t.Fatalf("device resources = %+v", resources)
	}
	commitDependencies := byName["commit"].Dependencies
	if len(commitDependencies) != 1 || commitDependencies[0] != "device" {
		t.Fatalf("commit dependencies = %v", commitDependencies)
	}
}
