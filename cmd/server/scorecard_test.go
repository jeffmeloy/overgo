package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
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

// TestScorecardHistoryReadsPublishedRecords: the launcher's history reader
// has nothing until two scorecards are published beside the store, then
// gives what moved between the newest two with which changes improved.
func TestScorecardHistoryReadsPublishedRecords(t *testing.T) {
	t.Parallel()
	storeRoot := filepath.Join(t.TempDir(), "overgodb-store")
	read := scorecardHistory(storeRoot)
	publish := func(stamp, commit string, throughput float64) {
		t.Helper()
		record := scorecard.Record{Commit: commit, Rows: []scorecard.RecordedRow{{Name: "alpha", Task: "inference",
			Cells: map[metric.Name]scorecard.RecordedCell{metric.Throughput: {Value: throughput, Unit: "tok/s"}}}}}
		if err := jsonfile.Write(filepath.Join(scorecard.RecordsDirectory(storeRoot), scorecard.RecordName(stamp)), record, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(scorecard.RecordsDirectory(storeRoot), 0o755); err != nil {
		t.Fatal(err)
	}
	publish("20260929T100000Z", "first", 40)
	if history, err := read(); err != nil || history != nil {
		t.Fatalf("one record gave %+v, %v", history, err)
	}
	publish("20260929T110000Z", "second", 45)
	history, err := read()
	if err != nil || history == nil || history.Previous != "first" || history.Current != "second" || len(history.Moves) != 1 ||
		history.Moves[0].Kind != string(scorecard.MoveChanged) || !history.Moves[0].Improved || history.Moves[0].Before != 40 || history.Moves[0].After != 45 {
		t.Fatalf("history = %+v, %v", history, err)
	}
	if scorecardHistory("") != nil {
		t.Fatal("a server without a store read a history")
	}
}
