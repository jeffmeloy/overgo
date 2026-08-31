package automationcheck

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func TestManifestExecutionBinding(t *testing.T) {
	plan, invocation := executionBindingFixture(t, strings.Repeat("b", 64))
	input, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("relevant inputs"))
	bound, err := BindManifestExecution(plan, invocation, []artifact.ID{input})
	if err != nil {
		t.Fatal(err)
	}
	if bound.ID == invocation.ID || bound.Authority == nil || bound.Authority.Definition != invocation.ID ||
		bound.Authority.Plan != plan.ID || !slices.Equal(bound.Authority.Inputs, []artifact.ID{input}) {
		t.Fatalf("bound invocation = %+v", bound)
	}
	otherPlan, otherInvocation := executionBindingFixture(t, strings.Repeat("c", 64))
	other, err := BindManifestExecution(otherPlan, otherInvocation, []artifact.ID{input})
	if err != nil || other.ID == bound.ID {
		t.Fatal("changed candidate tree did not change execution identity")
	}
}

func TestPlanEvidenceLineage(t *testing.T) {
	plan, invocation := executionBindingFixture(t, strings.Repeat("b", 64))
	input, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("relevant inputs"))
	bound, err := BindManifestExecution(plan, invocation, []artifact.ID{input})
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := Run(t.Context(), bound)
	if err != nil {
		t.Fatal(err)
	}
	lineage := EvidenceLineage(evidence)
	for _, want := range []artifact.ID{plan.ID, invocation.ID, input} {
		if !slices.Contains(lineage, want) {
			t.Fatalf("evidence lineage %v omits %s", lineage, want)
		}
	}
	if evidence.Authority == nil || evidence.Authority.Plan != plan.ID || evidence.Authority.Definition != invocation.ID || evidence.InvocationID != bound.ID {
		t.Fatalf("evidence authority = %+v", evidence)
	}
}

func executionBindingFixture(t *testing.T, candidateTree string) (ManifestPlan, Invocation) {
	t.Helper()
	checks := []Check{ownershipFixtureCheck("model", "owner:model", "internal/model")}
	surface := Surface{Identity: "base:candidate", Packages: []string{"internal/model"}}
	impact := OwnershipImpact(checks, surface)
	invocations, err := Plan(checks, impact)
	if err != nil {
		t.Fatal(err)
	}
	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate"))
	manifest, err := BindManifestPlan(base, candidate, strings.Repeat("a", 64), candidateTree, surface, impact, invocations)
	if err != nil {
		t.Fatal(err)
	}
	return manifest, invocations[0]
}
