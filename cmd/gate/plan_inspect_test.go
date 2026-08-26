package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

func TestPlanOutput(t *testing.T) {
	planned, _ := planInspectionFixture(t)
	var output bytes.Buffer
	if err := writeGatePlanReport(&output, planned); err != nil {
		t.Fatal(err)
	}
	var report gatePlanReport
	if err := json.Unmarshal(output.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Kind != "overgo.gate-plan-inspection" || report.Schema != 2 || report.PlanID != planned.manifest.ID.String() {
		t.Fatalf("report authority = %+v", report)
	}
	if report.BaseManifest == "" || report.CandidateManifest == "" || len(report.Selected) != 3 || len(report.Excluded) != 1 {
		t.Fatalf("report dispositions = %+v", report)
	}
	if !strings.Contains(output.String(), `"required_by": [`) || strings.Contains(output.String(), `"dependency_added"`) {
		t.Fatalf("dependency relationship output is not exact: %s", output.String())
	}
}

func TestPlanReasons(t *testing.T) {
	planned, _ := planInspectionFixture(t)
	report := buildGatePlanReport(planned)
	wantSelected := map[string]string{
		"always":     "always required by check definition",
		"triggered":  "matched derived impact facts",
		"unresolved": "impact analysis unresolved: reflection boundary",
	}
	for _, disposition := range report.Selected {
		if disposition.Reason != wantSelected[disposition.Name] {
			t.Errorf("selected %s reason = %q, want %q", disposition.Name, disposition.Reason, wantSelected[disposition.Name])
		}
	}
	if len(report.Unresolved) != 1 || report.Unresolved[0].Name != "unresolved" {
		t.Fatalf("unresolved = %+v", report.Unresolved)
	}
	if len(report.Excluded) != 1 || report.Excluded[0].Name != "excluded" || report.Excluded[0].Reason != "symbol closure is disjoint" {
		t.Fatalf("excluded = %+v", report.Excluded)
	}
}

func TestPlanDoesNotExecute(t *testing.T) {
	planned, executions := planInspectionFixture(t)
	var output bytes.Buffer
	if err := writeGatePlanReport(&output, planned); err != nil {
		t.Fatal(err)
	}
	if *executions != 0 {
		t.Fatalf("inspection executed %d check runners", *executions)
	}
}

func planInspectionFixture(t *testing.T) (plannedPipeline, *int) {
	t.Helper()
	executions := 0
	runner := func(context.Context, automationcheck.Invocation) (bool, string, error) {
		executions++
		return false, "", nil
	}
	checks := []automationcheck.Check{
		{Descriptor: automationcheck.Descriptor{Name: "always", Phase: runrecord.PhaseValidate, Always: true}, Run: runner},
		{Descriptor: automationcheck.Descriptor{
			Name: "triggered", Phase: runrecord.PhaseTest, Triggers: []automationcheck.Fact{"owner:triggered"},
			Inapplicable: "disjoint", Dependencies: []string{"always"},
		}, Run: runner},
		{Descriptor: automationcheck.Descriptor{
			Name: "unresolved", Phase: runrecord.PhaseBuild, Triggers: []automationcheck.Fact{"owner:unresolved"}, Inapplicable: "disjoint",
		}, Run: runner},
		{Descriptor: automationcheck.Descriptor{
			Name: "excluded", Phase: runrecord.PhaseVet, Triggers: []automationcheck.Fact{"owner:excluded"}, Inapplicable: "disjoint",
		}, Run: runner},
	}
	impact := automationcheck.Impact{
		Facts:      []automationcheck.Fact{"owner:triggered"},
		Exclusions: []automationcheck.Exclusion{{Check: "excluded", Reason: "symbol closure is disjoint"}},
	}
	invocations, err := automationcheck.Plan(checks, impact)
	if err != nil {
		t.Fatal(err)
	}
	base, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("base manifest"))
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate manifest"))
	if err != nil {
		t.Fatal(err)
	}
	surface := automationcheck.Surface{Identity: "surface", Unknown: []string{"reflection boundary"}}
	manifest, err := automationcheck.BindManifestPlan(base, candidate, strings.Repeat("a", 64), strings.Repeat("b", 64), surface, impact, invocations)
	if err != nil {
		t.Fatal(err)
	}
	return plannedPipeline{
		definitions: checks, invocations: invocations, impact: impact, surface: surface, manifest: &manifest,
	}, &executions
}
