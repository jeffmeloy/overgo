package automationcheck

import (
	"encoding/json"
	"fmt"
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

// TestPlanDigestsItsUnknowns holds a plan to binding its unresolved analysis
// by digest: the unknowns stay out of the stored plan however many there are,
// a different set binds a different plan, and a plan bound before the digest,
// holding its unknowns whole, still validates under its recorded identity.
func TestPlanDigestsItsUnknowns(t *testing.T) {
	checks := []Check{ownershipFixtureCheck("model", "owner:model", "internal/model")}
	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate"))
	bind := func(unknown ...string) ManifestPlan {
		t.Helper()
		surface := Surface{Identity: "base:candidate", Unknown: unknown}
		impact := OwnershipImpact(checks, surface)
		invocations, err := Plan(checks, impact)
		if err != nil {
			t.Fatal(err)
		}
		plan, err := BindManifestPlan(base, candidate, strings.Repeat("a", 64), strings.Repeat("b", 64), surface, impact, invocations)
		if err != nil {
			t.Fatal(err)
		}
		return plan
	}
	var many []string
	for symbol := range 300 {
		many = append(many, fmt.Sprintf("uncovered-symbol:overgo/reached:%d", symbol))
	}
	one, all := bind(many[0]), bind(many...)
	if len(all.Unknown) != 0 || all.UnknownCount != len(many) || !all.UnknownDigest.Valid() || all.ID == one.ID {
		t.Fatalf("plan stored %d unknowns, counted %d, digest %s; one and all share identity %v", len(all.Unknown), all.UnknownCount, all.UnknownDigest, all.ID == one.ID)
	}
	encoded, err := json.Marshal(all)
	if err != nil || strings.Contains(string(encoded), many[len(many)-1]) {
		t.Fatalf("stored plan names its unknowns: %v", err)
	}
	legacy := one
	legacy.UnknownDigest, legacy.UnknownCount, legacy.Unknown = artifact.ID{}, 0, []string{many[0]}
	if legacy.ID, err = manifestPlanID(legacy); err != nil {
		t.Fatal(err)
	}
	if err := legacy.Validate(); err != nil {
		t.Fatalf("a plan holding its unknowns whole no longer validates: %v", err)
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
