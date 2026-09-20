package repoanalysis

import (
	"path/filepath"
	"slices"
	"testing"
)

// TestStoreAccessCensus runs the census on the real repository and holds it
// to its contract: packages come most sites first, every total is the sum of
// its kinds, and the store's own tools are seen; then it overlays one
// package that reaches the store in every way and imports two domains'
// records, and holds the census to counting each kind once, ignoring the
// package's test file, and naming both domains -- so the numbers are
// computed, and a package that straddles domains says so. The one batch it
// counts is an empty literal that is assigned and then filled; the empty
// literal the package returns as a zero value builds nothing and is no site.
func TestStoreAccessCensus(t *testing.T) {
	snapshot, err := DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	census, err := StoreAccessCensus(snapshot)
	if err != nil || len(census) == 0 {
		t.Fatalf("census = %d packages, %v", len(census), err)
	}
	for index, access := range census {
		sum := 0
		for _, sites := range access.Sites {
			sum += sites
		}
		if sum != access.Total || access.Total == 0 || index > 0 && census[index-1].Total < access.Total {
			t.Fatalf("package %s: total %d, kinds sum %d, previous total %d", access.Package, access.Total, sum, census[max(index, 1)-1].Total)
		}
	}
	if !slices.ContainsFunc(census, func(access StoreAccess) bool { return access.Package == "internal/gate" && access.Sites["commit"] > 0 }) {
		t.Fatal("the census does not see the gate committing to the store")
	}

	const straddler = `package straddler

import (
	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

var _ = dataset.Path
var _ = runrecord.GateSchema

func touch(root string) {
	store, _ := overgodb.OpenReadOnly(root)
	batch := artifact.Batch{}
	batch.Key = "k"
	_, _ = store.Commit(nil, batch)
	_, _, _ = artifact.ResolveAlias(nil, store, "a")
	_, _, _ = artifact.ReadContent(nil, store, artifact.ID{})
	_, _ = store.Query(nil, overgodb.Query{})
}

func refuse() (artifact.Batch, error) { return artifact.Batch{}, nil }
`
	overlaid, err := snapshot.Overlay(map[string][]byte{
		"internal/straddler/straddler.go":      []byte(straddler),
		"internal/straddler/straddler_test.go": []byte(straddler),
	})
	if err != nil {
		t.Fatal(err)
	}
	census, err = StoreAccessCensus(overlaid)
	if err != nil {
		t.Fatal(err)
	}
	index := slices.IndexFunc(census, func(access StoreAccess) bool { return access.Package == "internal/straddler" })
	if index < 0 {
		t.Fatal("the overlaid package is absent from the census")
	}
	found := census[index]
	for _, kind := range []string{"open", "batch", "commit", "alias", "read", "query"} {
		if found.Sites[kind] != 1 {
			t.Fatalf("kind %s = %d sites in %+v, want one", kind, found.Sites[kind], found)
		}
	}
	if !slices.Equal(found.Domains, []string{"datasets", "evidence"}) {
		t.Fatalf("domains = %v, want the two whose records it imports", found.Domains)
	}
}
