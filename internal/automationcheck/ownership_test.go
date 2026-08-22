package automationcheck

import (
	"testing"

	"overgo/internal/runrecord"
)

func TestImpactOwnershipSelection(t *testing.T) {
	checks := []Check{
		ownershipFixtureCheck("first", "owner:first", "internal/first"),
		ownershipFixtureCheck("second", "owner:second", "internal/second"),
	}
	impact := OwnershipImpact(checks, Surface{
		Identity: "candidate", Packages: []string{"internal/second"},
	})
	planned, err := Plan(checks, impact)
	if err != nil || len(planned) != 1 || planned[0].Check.Name != "second" || len(impact.Exclusions) != 1 {
		t.Fatalf("ownership plan = %+v impact=%+v err=%v", planned, impact, err)
	}
	unknown := OwnershipImpact(checks, Surface{Identity: "candidate", Unknown: []string{"reflection"}})
	planned, err = Plan(checks, unknown)
	if err != nil || len(planned) != len(checks) || len(unknown.Exclusions) != 0 {
		t.Fatalf("unknown ownership plan = %+v impact=%+v err=%v", planned, unknown, err)
	}
}

func ownershipFixtureCheck(name string, fact Fact, packagePath string) Check {
	return Check{
		Descriptor: Descriptor{
			Name: name, Phase: runrecord.PhaseValidate,
			Triggers: []Fact{fact}, Inapplicable: "disjoint ownership",
			Ownership: Ownership{Fact: fact, Packages: []string{packagePath}},
		},
		Run: pass,
	}
}
