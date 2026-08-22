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

func TestImpactFalseNegativeCorpus(t *testing.T) {
	corpus := []struct {
		name      string
		ownership Ownership
		surface   Surface
	}{
		{
			name: "exact package", ownership: Ownership{Packages: []string{"internal/model"}},
			surface: Surface{Identity: "package-change", Packages: []string{"internal/model"}},
		},
		{
			name: "package prefix", ownership: Ownership{PackagePrefixes: []string{"internal/cuda"}},
			surface: Surface{Identity: "device-change", Packages: []string{"internal/cuda/executor"}},
		},
		{
			name: "exact symbol", ownership: Ownership{Symbols: []Symbol{{Package: "internal/model", Receiver: "Runtime", Name: "Run"}}},
			surface: Surface{Identity: "symbol-change", Symbols: []Symbol{{Package: "internal/model", Receiver: "Runtime", Name: "Run"}}},
		},
		{
			name: "unknown reflection boundary", ownership: Ownership{Packages: []string{"internal/model"}},
			surface: Surface{Identity: "unknown-change", Unknown: []string{"reflection:internal/model/runtime.go:Runtime.Run"}},
		},
	}
	for _, fixture := range corpus {
		t.Run(fixture.name, func(t *testing.T) {
			const fact Fact = "owner:required"
			fixture.ownership.Fact = fact
			check := Check{Descriptor: Descriptor{
				Name: "required", Phase: runrecord.PhaseValidate, Triggers: []Fact{fact},
				Inapplicable: "proven independent", Ownership: fixture.ownership,
			}, Run: pass}
			impact := OwnershipImpact([]Check{check}, fixture.surface)
			planned, err := Plan([]Check{check}, impact)
			if err != nil || len(planned) != 1 {
				t.Fatalf("known-required check excluded: impact=%+v planned=%+v err=%v", impact, planned, err)
			}
			if _, excluded := impact.ExclusionReason("required"); excluded {
				t.Fatalf("known-required check has exclusion: %+v", impact)
			}
		})
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
