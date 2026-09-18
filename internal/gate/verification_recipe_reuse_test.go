package gate

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

// TestVerificationRecipeReuseAcceptance proves a successful verification node is
// reused across different recipe compositions instead of re-executing. Two
// manifest plans with different candidate trees give the same node different
// bound invocation ids; the first composition executes and records the node, the
// second reuses that evidence under its own authority citing the original, so the
// node runs once rather than once per recipe. A changed input still re-executes.
func TestVerificationRecipeReuseAcceptance(t *testing.T) {
	t.Parallel()
	runs := 0
	check := automationcheck.Check{
		Descriptor: automationcheck.Descriptor{Name: "node", Phase: runrecord.PhaseValidate, Always: true},
		Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
			runs++
			return false, "", nil
		},
	}
	invocations, err := automationcheck.Plan([]automationcheck.Check{check}, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}

	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	source := strings.Repeat("a", 64)
	surface := automationcheck.Surface{Identity: "base:candidate"}
	manifest := func(tree string) automationcheck.ManifestPlan {
		candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate-"+tree))
		plan, err := automationcheck.BindManifestPlan(base, candidate, source, tree, surface, automationcheck.Impact{}, invocations)
		if err != nil {
			t.Fatal(err)
		}
		return plan
	}
	planA, planB := manifest(strings.Repeat("b", 64)), manifest(strings.Repeat("c", 64))
	if planA.ID == planB.ID {
		t.Fatal("distinct recipe compositions share a plan identity")
	}

	environment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("environment"))
	input, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("node inputs"))
	bind := func(plan automationcheck.ManifestPlan, in artifact.ID) automationcheck.Invocation {
		bound, err := automationcheck.BindManifestExecution(plan, invocations[0], []artifact.ID{in},
			&automationcheck.ReuseBinding{Input: in, Environment: environment})
		if err != nil {
			t.Fatal(err)
		}
		return bound
	}
	boundA, boundB := bind(planA, input), bind(planB, input)
	if boundA.ID == boundB.ID {
		t.Fatal("the same node bound identically across distinct recipes")
	}

	cache := automationcheck.NewEvidenceCache(environment)
	first, reused, err := cache.RunCached(t.Context(), boundA, input)
	if err != nil || reused || runs != 1 {
		t.Fatalf("first composition = (reused=%t, %v) runs=%d", reused, err, runs)
	}
	second, reused, err := cache.RunCached(t.Context(), boundB, input)
	if err != nil || !reused || runs != 1 || !second.Reused || second.Source == nil ||
		second.Source.Evidence != first.ID || second.ID == first.ID || second.VerifyIdentity() != nil {
		t.Fatalf("second composition did not reuse the node: (%+v, %t, %v) runs=%d", second, reused, err, runs)
	}
	if err := automationcheck.ValidateReuseAuthority(second, invocations[0].ID); err != nil {
		t.Fatalf("reused node failed authority validation: %v", err)
	}

	// A changed input re-executes even under a matching recipe node, so reuse
	// never launders a stale input into a pass.
	other, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("changed inputs"))
	if _, reused, err := cache.RunCached(t.Context(), bind(planB, other), other); err != nil || reused || runs != 2 {
		t.Fatalf("changed input reused a node: (reused=%t, %v) runs=%d", reused, err, runs)
	}
}
