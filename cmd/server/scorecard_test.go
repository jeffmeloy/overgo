package main

import (
	"testing"

	"overgo/internal/overgodb"
)

// TestServerScorecardReadsTheStore: the launcher's scorecard reader builds
// the scorecard from a real store; one with no activated model scores none.
func TestServerScorecardReadsTheStore(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	rows, err := serverScorecard(t.Context(), store)
	if err != nil || len(rows) != 0 {
		t.Fatalf("an empty store scored %+v, %v", rows, err)
	}
}
