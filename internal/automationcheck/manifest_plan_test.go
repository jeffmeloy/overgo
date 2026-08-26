package automationcheck

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func TestManifestPlanBindsCandidateAndDefinitions(t *testing.T) {
	checks := []Check{ownershipFixtureCheck("model", "owner:model", "internal/model")}
	surface := Surface{Identity: "base:candidate", Packages: []string{"internal/model"}}
	impact := OwnershipImpact(checks, surface)
	invocations, err := Plan(checks, impact)
	if err != nil {
		t.Fatal(err)
	}
	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate"))
	bound, err := BindManifestPlan(base, candidate, strings.Repeat("a", 64), strings.Repeat("b", 64), surface, impact, invocations)
	if err != nil {
		t.Fatal(err)
	}
	if err := bound.Validate(); err != nil {
		t.Fatal(err)
	}
	again, err := BindManifestPlan(base, candidate, strings.Repeat("a", 64), strings.Repeat("b", 64), surface, impact, invocations)
	if err != nil || again.ID != bound.ID {
		t.Fatalf("plan identity is unstable: %s %s %v", bound.ID, again.ID, err)
	}
	other, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("other candidate"))
	changed, err := BindManifestPlan(base, other, strings.Repeat("a", 64), strings.Repeat("b", 64), surface, impact, invocations)
	if err != nil || changed.ID == bound.ID {
		t.Fatal("candidate identity did not bind the manifest plan")
	}
}

func TestUnknownRunsChecksInManifestPlan(t *testing.T) {
	checks := []Check{ownershipFixtureCheck("model", "owner:model", "internal/model")}
	surface := Surface{Identity: "base:candidate", Unknown: []string{"reflection"}}
	impact := OwnershipImpact(checks, surface)
	invocations, err := Plan(checks, impact)
	if err != nil || len(invocations) != 1 {
		t.Fatalf("unknown plan = %+v, %v", invocations, err)
	}
}

func TestExclusionRequiresProofInManifestPlan(t *testing.T) {
	checks := []Check{ownershipFixtureCheck("model", "owner:model", "internal/model")}
	if _, err := Plan(checks, Impact{Exclusions: []Exclusion{{Check: "model"}}}); err == nil {
		t.Fatal("reasonless exclusion was accepted")
	}
}
