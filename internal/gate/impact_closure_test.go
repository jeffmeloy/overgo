package gate

import (
	"slices"
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
)

// TestOwnershipClosureExcludesUntouchedLanes pins the linker's rule under
// symbol uncertainty: interface, reflection and cgo uncertainties alone admit
// the dependency fallback while a non-Go or build-selection uncertainty
// keeps the complete plan; the changed packages are the seeds and the
// owners of changed external inputs; over the compiler fixture's live
// import graph a lane owning an untouched package is excluded, a lane whose
// owned package reaches the changed package runs, and a symbol-owned check
// is never excluded by dependency alone.
func TestOwnershipClosureExcludesUntouchedLanes(t *testing.T) {
	t.Parallel()
	symbolOnly := codemanifest.Impact{Uncertainty: []codemanifest.Uncertainty{
		{Kind: codemanifest.UncertaintyInterface}, {Kind: codemanifest.UncertaintyReflection}, {Kind: codemanifest.UncertaintyCgo},
	}}
	if !reachabilityOnlyUncertainty(symbolOnly) {
		t.Fatal("symbol-only uncertainty refused the dependency fallback")
	}
	for _, kind := range []codemanifest.UncertaintyKind{codemanifest.UncertaintyNonGo, codemanifest.UncertaintyBuildSelection, codemanifest.UncertaintyOutsideSnapshot} {
		mixed := codemanifest.Impact{Uncertainty: append(slices.Clone(symbolOnly.Uncertainty), codemanifest.Uncertainty{Kind: kind})}
		if reachabilityOnlyUncertainty(mixed) {
			t.Fatalf("%s uncertainty admitted the dependency fallback", kind)
		}
	}
	if reachabilityOnlyUncertainty(codemanifest.Impact{}) {
		t.Fatal("a resolved impact admitted the fallback it does not need")
	}
	boundaries := codemanifest.Impact{Uncertainty: []codemanifest.Uncertainty{
		{Kind: codemanifest.UncertaintyExternal, Reason: consumerBoundaryReason + "command"},
		{Kind: codemanifest.UncertaintyExternal, Reason: consumerBoundaryReason + "method-dispatch"},
	}}
	if !reachabilityOnlyUncertainty(boundaries) {
		t.Fatal("consumer census boundaries refused the dependency fallback")
	}
	if reachabilityOnlyUncertainty(codemanifest.Impact{Uncertainty: []codemanifest.Uncertainty{{Kind: codemanifest.UncertaintyExternal, Reason: "exported across the snapshot"}}}) {
		t.Fatal("an external uncertainty outside the consumer census admitted the fallback")
	}

	seeded := codemanifest.Impact{
		Seeds: []codemanifest.SymbolID{{Package: "internal/recipe", Name: "Value"}, {Package: "internal/recipe", Name: "Other"}},
		ExternalInputs: []codemanifest.ExternalInputChange{
			{Path: "internal/plan/catalog.txt", Candidate: &codemanifest.ExternalInput{Owner: "internal/plan"}},
		},
	}
	if changed := changedPackages(seeded); !slices.Equal(changed, []string{"internal/plan", "internal/recipe"}) {
		t.Fatalf("changed packages = %v", changed)
	}

	g := scopeCompilerFixture(t)
	resolver, err := g.dependencyResolver()
	if err != nil {
		t.Fatal(err)
	}
	owning := func(name string, fact automationcheck.Fact, packages ...string) automationcheck.Check {
		return automationcheck.Check{Descriptor: automationcheck.Descriptor{Name: name, Ownership: automationcheck.Ownership{Fact: fact, Packages: packages}}}
	}
	symbolOwned := automationcheck.Check{Descriptor: automationcheck.Descriptor{Name: "symbols", Ownership: automationcheck.Ownership{
		Fact: "symbol fact", Symbols: []automationcheck.Symbol{{Package: "internal/other", Name: "X"}},
	}}}
	checks := []automationcheck.Check{owning("client-lane", "client fact", "internal/client"), owning("other-lane", "other fact", "internal/other"), symbolOwned}

	impact := automationcheck.OwnershipByDependency(checks, []string{"internal/recipe"}, resolver)
	excluded := func(impact automationcheck.Impact) []string {
		var names []string
		for _, exclusion := range impact.Exclusions {
			names = append(names, exclusion.Check)
		}
		return names
	}
	if !slices.Equal(excluded(impact), []string{"other-lane"}) || !slices.Equal(impact.Facts, []automationcheck.Fact{"client fact"}) {
		t.Fatalf("recipe change: excluded=%v facts=%v", excluded(impact), impact.Facts)
	}
	impact = automationcheck.OwnershipByDependency(checks, []string{"internal/other"}, resolver)
	if !slices.Equal(excluded(impact), []string{"client-lane"}) || !slices.Equal(impact.Facts, []automationcheck.Fact{"other fact"}) {
		t.Fatalf("other change: excluded=%v facts=%v", excluded(impact), impact.Facts)
	}
	if impact := automationcheck.OwnershipByDependency(checks, []string{"internal/other"}, nil); len(impact.Exclusions) != 0 {
		t.Fatalf("a nil resolver excluded %v", excluded(impact))
	}
}
