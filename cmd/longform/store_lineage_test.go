package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// requireStoreLineage skips a store-bound guard test when the store under
// OVERGO_DATA_ROOT does not carry the bound record: the exact control
// records exist only in the producer's store lineage, and another store
// (a lane store forked before they were recorded) cannot check them.
func requireStoreLineage(t *testing.T, store *overgodb.Store, text string) {
	t.Helper()
	id, err := artifact.ParseID(text)
	if err != nil {
		t.Fatal(err)
	}
	found, err := store.HasContent(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Skipf("integration: the bound guard record %s is absent from this store; only the producer's store lineage carries it", id)
	}
}
