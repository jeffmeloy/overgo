package gate

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestGateOrdersWorkByMeasuredCost holds a gate's test runs to starting the
// packages the last recorded suite cost measured longest first, with the
// unmeasured ones ahead of them, and to keeping the order when the store has
// no recorded cost.
func TestGateOrdersWorkByMeasuredCost(t *testing.T) {
	t.Parallel()
	packages := []string{"overgo/cmd/a", "overgo/cmd/b", "overgo/internal/gate", "overgo/internal/new"}
	storePath := filepath.Join(t.TempDir(), "store")
	if got := (&gateContext{storePath: storePath}).orderByMeasuredCost(packages); !slices.Equal(got, packages) {
		t.Fatalf("with no recorded cost the order moved: %v", got)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	content, err := suiteCostContract.ContentBytes([]byte(`{"invocations": [
		{"executions": [{"package": "overgo/cmd/a", "elapsed_seconds": 3}, {"package": "overgo/internal/gate", "elapsed_seconds": 146}]},
		{"executions": [{"package": "overgo/cmd/b", "elapsed_seconds": 9}, {"package": "overgo/cmd/a", "elapsed_seconds": 12}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.CommitAs(t.Context(), gateProducer, artifact.Batch{
		Key: "test/suite-cost", Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{{Name: suiteCostAlias, Target: content.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	want := []string{"overgo/internal/new", "overgo/internal/gate", "overgo/cmd/a", "overgo/cmd/b"}
	if got := (&gateContext{storePath: storePath}).orderByMeasuredCost(packages); !slices.Equal(got, want) {
		t.Fatalf("measured order = %v, want %v", got, want)
	}
}
