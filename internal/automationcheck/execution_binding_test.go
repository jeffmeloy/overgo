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
	bound, err := BindManifestExecution(plan, invocation, []artifact.ID{input}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if bound.ID == invocation.ID || bound.Authority == nil || bound.Authority.Definition != invocation.ID ||
		bound.Authority.Plan != plan.ID || !slices.Equal(bound.Authority.Inputs, []artifact.ID{input}) {
		t.Fatalf("bound invocation = %+v", bound)
	}
	otherPlan, otherInvocation := executionBindingFixture(t, strings.Repeat("c", 64))
	other, err := BindManifestExecution(otherPlan, otherInvocation, []artifact.ID{input}, nil)
	if err != nil || other.ID == bound.ID {
		t.Fatal("changed candidate tree did not change execution identity")
	}
	environment, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("execution environment"))
	binding := ReuseBinding{Input: input, Environment: environment}
	reusable, err := BindManifestExecution(plan, invocation, []artifact.ID{input}, &binding)
	if err != nil || reusable.ID == bound.ID {
		t.Fatalf("reuse binding did not change execution identity: %v", err)
	}
	binding.Input, binding.Environment = binding.Environment, binding.Input
	if reusable.Authority.Reuse.Input != input || reusable.Authority.Reuse.Environment != environment {
		t.Fatal("caller mutation changed bound execution inputs")
	}
	swapped, err := BindManifestExecution(plan, invocation, []artifact.ID{input}, &binding)
	if err != nil || swapped.ID == reusable.ID {
		t.Fatalf("input/environment roles were lost: %v", err)
	}
	if _, err := BindManifestExecution(plan, invocation, []artifact.ID{input}, &ReuseBinding{Input: input}); err == nil {
		t.Fatal("incomplete reuse binding admitted")
	}
}

func TestPlanEvidenceLineage(t *testing.T) {
	plan, invocation := executionBindingFixture(t, strings.Repeat("b", 64))
	input, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("relevant inputs"))
	memo, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("memo inputs"))
	environment, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("environment"))
	bound, err := BindManifestExecution(plan, invocation, []artifact.ID{input}, &ReuseBinding{Input: memo, Environment: environment})
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := Run(t.Context(), bound)
	if err != nil {
		t.Fatal(err)
	}
	lineage := EvidenceLineage(evidence)
	for _, want := range []artifact.ID{plan.ID, invocation.ID, input, memo, environment} {
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
