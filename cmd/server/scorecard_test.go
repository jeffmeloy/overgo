package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
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
// gives what moved between the newest two, a faster throughput as better,
// and refuses a record the scorecard refuses rather than reading around it.
func TestScorecardHistoryReadsPublishedRecords(t *testing.T) {
	t.Parallel()
	storeRoot := filepath.Join(t.TempDir(), "overgodb-store")
	read := scorecardHistory(storeRoot)
	model := testutil.ArtifactID(t, artifact.KindModel, "alpha")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "measured")
	published := time.Date(2026, time.September, 29, 10, 0, 0, 0, time.UTC)
	publish := func(commit string, cell scorecard.Cell) {
		t.Helper()
		cells := map[metric.Name]scorecard.Cell{}
		for _, column := range metric.Columns {
			cells[column] = scorecard.Cell{Reason: "no record measures it"}
		}
		cells[metric.Throughput] = cell
		record := scorecard.Record{Commit: commit, Rows: []scorecard.Row{{Model: model, Name: "alpha", Task: recipe.TaskInference, Cells: cells}}}
		published = published.Add(time.Hour)
		if _, err := scorecard.Publish(scorecard.Directory(storeRoot), published, record); err != nil {
			t.Fatal(err)
		}
	}
	throughput := func(value float64) scorecard.Cell {
		return scorecard.Cell{Value: value, Unit: "tok/s", Source: evidence, Contract: scorecard.Contract{Metric: "throughput", Origin: "serving"}}
	}
	publish("first", throughput(40))
	if history, err := read(); err != nil || history != nil {
		t.Fatalf("one record gave %+v, %v", history, err)
	}
	publish("second", throughput(45))
	history, err := read()
	if err != nil || history == nil || history.Previous != "first" || history.Current != "second" || len(history.Moves) != 1 ||
		history.Moves[0].Kind != "better" || history.Moves[0].Before != 40 || history.Moves[0].After != 45 {
		t.Fatalf("history = %+v, %v", history, err)
	}
	// A record from before the scorecard's own rules (a cell nothing explains) is refused.
	refused := filepath.Join(scorecard.Directory(storeRoot), "scorecard-"+published.Add(time.Hour).UTC().Format("20060102T150405Z")+".json")
	if err := os.WriteFile(refused, []byte(`{"commit":"third","rows":[{"name":"alpha","task":"inference","cells":{"throughput":{"value":1}}}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if history, err := read(); err == nil || history != nil {
		t.Fatalf("a record with an unexplained cell read as %+v, %v", history, err)
	}
	if scorecardHistory("") != nil {
		t.Fatal("a server without a store read a history")
	}
}
