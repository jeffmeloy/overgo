package server

import (
	"context"
	"encoding/json"
	"net/http"
	"reflect"
	"testing"

	"overgo/internal/overgodb"
)

// testScorecardHistory is what moved between two published scorecards: a
// faster throughput, a slower first output, a newly measured quality, a lost
// peak memory, a new model and a gone one.
func testScorecardHistory() *ScorecardHistory {
	return &ScorecardHistory{Previous: "0123456789abcdef", Current: "fedcba9876543210", Moves: []ScorecardMove{
		{Name: "gemma-e4b", Task: "inference", Kind: "changed", Metric: "throughput", Before: 40, After: 45, Unit: "tok/s", Improved: true},
		{Name: "gemma-e4b", Task: "inference", Kind: "changed", Metric: "first-output", Before: 100, After: 130, Unit: "ms"},
		{Name: "gemma-e4b", Task: "inference", Kind: "measured", Metric: "held-out-quality", After: 0.7},
		{Name: "gemma-e4b", Task: "inference", Kind: "unmeasured", Metric: "peak-device-memory", Reason: "no benchmark record"},
		{Name: "qwen-0.5b", Task: "inference", Kind: "added"},
		{Name: "retired", Task: "inference", Kind: "removed"},
	}}
}

// TestWhatGotBetterBetweenVersions: the scorecard route carries what moved
// between the two newest published scorecards when the launcher reads them,
// and no history before two are published or without the reader.
func TestWhatGotBetterBetweenVersions(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	read := func(history ScorecardHistoryReader) scorecardResponse {
		t.Helper()
		handler := newTestHandlerForRepository(t, store, &fakeGenerator{}, func(config *Config) {
			config.Scorecard = func(context.Context, *overgodb.Store) ([]ScorecardRow, error) { return nil, nil }
			config.ScorecardHistory = history
		})
		response := serveTestRequest(handler, http.MethodGet, "/scorecard", "")
		var decoded scorecardResponse
		if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil || response.Code != http.StatusOK {
			t.Fatalf("scorecard %d %s: %v", response.Code, response.Body.String(), err)
		}
		return decoded
	}
	if got := read(func() (*ScorecardHistory, error) { return testScorecardHistory(), nil }); !reflect.DeepEqual(got.History, testScorecardHistory()) {
		t.Fatalf("history = %+v", got.History)
	}
	if got := read(func() (*ScorecardHistory, error) { return nil, nil }); got.History != nil {
		t.Fatalf("one published scorecard carried history %+v", got.History)
	}
	if got := read(nil); got.History != nil {
		t.Fatalf("a server without the history reader carried %+v", got.History)
	}
}
