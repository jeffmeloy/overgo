package server

import (
	"context"
	"encoding/json"
	"net/http"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// testScorecard is a measured inference row and a training row whose first
// output no record measures.
func testScorecard(t *testing.T) []ScorecardRow {
	t.Helper()
	model := testutil.ArtifactBytesID(t, artifact.KindModel, []byte("gemma-e4b"))
	return []ScorecardRow{
		{Model: model, Name: "gemma-e4b", Task: "inference", Cells: []ScorecardCell{
			{Metric: "first-output", Value: 12.34, Unit: "ms", Label: "benchmark time to first token, median"},
			{Metric: "throughput", Value: 45.67, Unit: "tok/s", Label: "benchmark decode rate, median"},
			{Metric: "held-out-quality", Value: 0.8125, Label: "accuracy, arc (zero-shot)"},
			{Metric: "failure-rate", Value: 0.025, Label: "1 of 40 attempts failed, 0 cancelled"},
			{Metric: "load-time", Value: 1520, Unit: "ms", Label: "benchmark load"},
			{Metric: "peak-device-memory", Value: 1.5 * (1 << 30), Unit: "bytes", Label: "benchmark peak device memory"},
			{Metric: "setup-steps", Value: 3, Label: "lifecycle transitions to first activation"},
		}},
		{Model: model, Name: "gemma-e4b", Task: "training", Cells: []ScorecardCell{
			{Metric: "first-output", Reason: "a training session has no first output"},
		}},
	}
}

// TestScorecardView: the scorecard route answers that it needs the
// launcher's reader when it has none, and otherwise the rows the reader
// builds from the server's store, an empty scorecard as an empty list.
func TestScorecardView(t *testing.T) {
	t.Parallel()
	if response := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/scorecard", ""); response.Code != http.StatusNotFound {
		t.Fatalf("a server without a scorecard reader answered %d", response.Code)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	read := func(rows []ScorecardRow) (int, string, scorecardResponse) {
		t.Helper()
		handler := newTestHandlerForRepository(t, store, &fakeGenerator{}, func(config *Config) {
			config.Scorecard = func(_ context.Context, given *overgodb.Store) ([]ScorecardRow, error) {
				if given != store {
					t.Error("the reader was not given the server's store")
				}
				return rows, nil
			}
		})
		response := serveTestRequest(handler, http.MethodGet, "/scorecard", "")
		var decoded scorecardResponse
		_ = json.Unmarshal(response.Body.Bytes(), &decoded)
		return response.Code, response.Body.String(), decoded
	}
	if code, _, got := read(testScorecard(t)); code != http.StatusOK || !reflect.DeepEqual(got.Rows, testScorecard(t)) {
		t.Fatalf("scorecard %d %+v", code, got)
	}
	if code, body, _ := read(nil); code != http.StatusOK || body != "{\"rows\":[]}\n" {
		t.Fatalf("an empty scorecard answered %d %q", code, body)
	}
}
