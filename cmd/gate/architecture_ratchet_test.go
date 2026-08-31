package main

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

func TestArchitectureRatchetAlwaysRequired(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	gate := &gateContext{repo: repo, paths: []string{"docs/plan.json"}}
	checks := gate.pipelineChecks()
	positions := make(map[string]int, len(checks))
	var architecture automationcheck.Check
	count := 0
	for index, check := range checks {
		positions[check.Descriptor.Name] = index
		if check.Descriptor.Name == "architecture" {
			architecture, count = check, count+1
		}
	}
	if count != 1 {
		t.Fatalf("architecture check count = %d", count)
	}
	descriptor := architecture.Descriptor
	if !descriptor.Always || descriptor.Phase != runrecord.PhaseValidate ||
		len(descriptor.Triggers) != 0 || descriptor.Inapplicable != "" ||
		descriptor.Ownership.Fact != "" || len(descriptor.Ownership.Packages) != 0 ||
		len(descriptor.Ownership.PackagePrefixes) != 0 || len(descriptor.Ownership.Symbols) != 0 {
		t.Fatalf("architecture descriptor is not ownership-free and always required: %+v", descriptor)
	}
	if !slices.Equal(descriptor.Dependencies, []string{"scope"}) ||
		!slices.Equal(checks[positions["profile"]].Descriptor.Dependencies, []string{"architecture"}) ||
		positions["scope"] >= positions["architecture"] || positions["architecture"] >= positions["profile"] {
		t.Fatalf("architecture dependency order = scope:%d architecture:%d profile:%d", positions["scope"], positions["architecture"], positions["profile"])
	}
	for _, expensive := range []string{"acceptance", "vet", "build", "test", "device"} {
		if positions["architecture"] >= positions[expensive] {
			t.Fatalf("architecture ratchet at %d follows %s at %d", positions["architecture"], expensive, positions[expensive])
		}
	}

	// A documentation-only impact excludes ownership-selected checks, but the
	// architecture ratchet remains selected because it owns no exclusion path.
	surface := automationcheck.Surface{Identity: "documentation-only"}
	impact := automationcheck.OwnershipImpact(checks, surface)
	invocations, err := automationcheck.Plan(checks, impact)
	if err != nil {
		t.Fatal(err)
	}
	var architectureInvocation automationcheck.Invocation
	for _, invocation := range invocations {
		if invocation.Check.Name == "architecture" {
			architectureInvocation = invocation
			break
		}
	}
	if !architectureInvocation.ID.Valid() {
		t.Fatal("documentation-only plan omitted the architecture ratchet")
	}
	evidence, runErr := automationcheck.Run(t.Context(), architectureInvocation)
	if runErr != nil {
		t.Fatal(runErr)
	}
	if evidence.Inapplicable || evidence.DurationNS == 0 {
		t.Fatalf("architecture evidence = %+v", evidence)
	}
	if !slices.ContainsFunc(gate.honesty, func(line string) bool {
		return strings.HasPrefix(line, "entry authority ratchet:") && strings.Contains(line, "wall=")
	}) {
		t.Fatalf("architecture ratchet omitted measured entry-authority evidence: %q", gate.honesty)
	}

	base, err := artifact.JSONID(artifact.KindProfile, struct{ Name string }{"base"})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := artifact.JSONID(artifact.KindProfile, struct{ Name string }{"candidate"})
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := gate.sourceSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := automationcheck.BindManifestPlan(
		base, candidate, snapshot.Identity(), strings.Repeat("0", 64), surface, impact, invocations,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(manifest.Invocations, func(invocation automationcheck.PlannedInvocation) bool {
		return invocation.Check.Name == "architecture" && invocation.ID == architectureInvocation.ID
	}) {
		t.Fatal("manifest plan did not bind the architecture invocation")
	}
}

func TestProductionAuthorityBoundaries(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	clean, err := repoanalysis.AuditProductionAuthorityBoundaries(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if err := clean.Error(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		family  string
		overlay map[string][]byte
	}{
		{name: "storage", family: "storage", overlay: map[string][]byte{
			"internal/server/rogue_commit.go": []byte("package server\nfunc rogueCommit(value interface{ Commit() error }) { _ = value.Commit() }\n"),
		}},
		{name: "aliased process", family: "process", overlay: map[string][]byte{
			"internal/server/rogue_process.go": []byte("package server\nimport runner \"os/exec\"\nfunc rogueProcess() { _ = runner.Command(\"probe\") }\n"),
		}},
		{name: "capability", family: "capability", overlay: map[string][]byte{
			"internal/server/rogue_capability.go": []byte("package server\nimport capability \"overgo/internal/capabilityruntime\"\nfunc rogueCapability(c capability.ExecutorCatalog) { _ = c.Execute() }\n"),
		}},
		{name: "tool", family: "tool", overlay: map[string][]byte{
			"internal/server/rogue_tool.go": []byte("package server\nimport tools \"overgo/internal/agenttool\"\nfunc rogueTool(e *tools.Executor) { _, _ = e.Invoke(nil, tools.Manual{}, nil) }\n"),
		}},
		{name: "trigger", family: "trigger", overlay: map[string][]byte{
			"internal/server/rogue_trigger.go": []byte("package server\nimport records \"overgo/internal/runrecord\"\nvar rogueTrigger = records.CausalContext{Trigger: records.TriggerWebhook}\n"),
		}},
		{name: "promotion", family: "promotion", overlay: map[string][]byte{
			"internal/server/rogue_promotion.go": []byte("package server\nimport comp \"overgo/internal/composition\"\nfunc roguePromotion() { _, _, _, _ = comp.ActiveCompositeGeneration(nil, nil, artifact.ID{}, artifact.ID{}, recipe.Task{}) }\n"),
		}},
		{name: "missing owner", family: "promotion", overlay: map[string][]byte{
			"internal/inference/composite_generation_surface.go": nil,
		}},
		{name: "stale allowance", family: "process", overlay: map[string][]byte{
			"cmd/eval-lane/main.go": nil,
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate, err := snapshot.Overlay(test.overlay)
			if err != nil {
				t.Fatal(err)
			}
			report, err := repoanalysis.AuditProductionAuthorityBoundaries(candidate)
			if err != nil {
				t.Fatal(err)
			}
			if report.Error() == nil {
				t.Fatalf("%s bypass was admitted", test.family)
			}
			if !slices.ContainsFunc(report.Findings, func(finding repoanalysis.ProductionAuthorityFinding) bool {
				return finding.Family == test.family
			}) {
				t.Fatalf("%s bypass produced findings for the wrong family: %+v", test.family, report.Findings)
			}
		})
	}
}
