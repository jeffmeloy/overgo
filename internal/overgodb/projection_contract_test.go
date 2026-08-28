package overgodb

import (
	"reflect"
	"testing"
)

// TestProjectionContractIsClosedAndVersioned holds the projection
// registry to its contract: every facet field of catalogState is
// enumerated exactly once, names are unique and explicit, every
// version starts at the shared origin, and registry-driven
// application is the one apply path (the facet equivalence and
// coordinator tests prove its behavior). Reflection appears only
// here, to prove the compiled registry is closed over the aggregate.
func TestProjectionContractIsClosedAndVersioned(t *testing.T) {
	state := newCatalogState()
	views := projections(&state)
	facetFields := reflect.TypeOf(state).NumField()
	if len(views) != facetFields {
		t.Fatalf("registry enumerates %d projections; catalogState composes %d facets", len(views), facetFields)
	}
	names := map[string]bool{}
	for _, registered := range views {
		if registered.name == "" || names[registered.name] {
			t.Fatalf("projection name %q is empty or repeated", registered.name)
		}
		names[registered.name] = true
		if registered.version != initialProjectionVersion {
			t.Fatalf("projection %q version %d; every projection starts at %d",
				registered.name, registered.version, initialProjectionVersion)
		}
		if registered.view == nil {
			t.Fatalf("projection %q has no compiled view", registered.name)
		}
	}
	for _, required := range []string{"artifacts", "contents", "lineage", "locations", "aliases", "commits"} {
		if !names[required] {
			t.Fatalf("projection %q missing from the closed registry", required)
		}
	}
}
