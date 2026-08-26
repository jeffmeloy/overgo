package automationcheck

import "testing"

func TestOwnershipCompletenessMakesUncoveredSurfaceUnknown(t *testing.T) {
	checks := []Check{ownershipFixtureCheck("model", "owner:model", "internal/model")}
	effective, surface, coverage, err := CompleteOwnership(checks, Surface{
		Identity: "candidate", Packages: []string{"internal/new"},
		Symbols: []Symbol{{Package: "internal/new", Name: "Run"}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(effective) != 1 || len(coverage.UncoveredPackages) != 1 || len(coverage.UncoveredSymbols) != 1 || len(surface.Unknown) != 2 {
		t.Fatalf("surface=%+v coverage=%+v", surface, coverage)
	}
	if impact := OwnershipImpact(effective, surface); len(impact.Exclusions) != 0 {
		t.Fatalf("uncovered surface authorized exclusions: %+v", impact)
	}
}

func TestContradictoryOwnershipFactIsRejected(t *testing.T) {
	checks := []Check{
		ownershipFixtureCheck("first", "owner:shared", "internal/first"),
		ownershipFixtureCheck("second", "owner:shared", "internal/second"),
	}
	if _, _, _, err := CompleteOwnership(checks, Surface{Identity: "candidate"}, nil); err == nil {
		t.Fatal("contradictory fact ownership was accepted")
	}
}

func TestReviewedOverlayIsSeparateAndEffective(t *testing.T) {
	checks := []Check{ownershipFixtureCheck("model", "owner:model", "internal/model")}
	effective, surface, coverage, err := CompleteOwnership(checks, Surface{
		Identity: "candidate", Packages: []string{"internal/legacy"},
	}, []ReviewedOwnership{{Check: "model", Reason: "legacy generator boundary", Packages: []string{"internal/legacy"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(coverage.UncoveredPackages) != 0 || len(surface.Unknown) != 0 || len(effective[0].Descriptor.Ownership.Packages) != 2 {
		t.Fatalf("surface=%+v coverage=%+v effective=%+v", surface, coverage, effective[0].Descriptor.Ownership)
	}
	if len(checks[0].Descriptor.Ownership.Packages) != 1 {
		t.Fatal("reviewed overlay mutated the original descriptor")
	}
}
