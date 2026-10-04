package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

// testScorecardHistory is what moved between two published scorecards: a
// faster throughput, a slower first output, a newly measured quality, a lost
// peak memory, a load time measured differently, a setup count that changed
// with no direction, a new model and a gone one.
func testScorecardHistory() *ScorecardHistory {
	return &ScorecardHistory{Previous: "0123456789abcdef", Current: "fedcba9876543210", Moves: []ScorecardMove{
		{Name: "gemma-e4b", Task: "inference", Kind: "better", Metric: "throughput", Before: 40, After: 45, Unit: "tok/s"},
		{Name: "gemma-e4b", Task: "inference", Kind: "worse", Metric: "first-output", Before: 100, After: 130, Unit: "ms"},
		{Name: "gemma-e4b", Task: "inference", Kind: "changed", Metric: "setup-steps", Before: 2, After: 3},
		{Name: "gemma-e4b", Task: "inference", Kind: "measured", Metric: "held-out-quality", After: 0.7},
		{Name: "gemma-e4b", Task: "inference", Kind: "unmeasured", Metric: "peak-device-memory", Reason: "no benchmark record"},
		{Name: "gemma-e4b", Task: "inference", Kind: "unknown", Metric: "load-time", Reason: "measured another way than the baseline"},
		{Name: "qwen-0.5b", Task: "inference", Kind: "added"},
		{Name: "retired", Task: "inference", Kind: "removed"},
	}}
}

// TestScorecardMoveKeepsAZeroMeasure holds a move's before and after to the
// JSON the page reads even at zero, the healthy baseline of a failure rate and
// the peak memory of a host run, which the page would otherwise read as absent
// and print as not a number.
func TestScorecardMoveKeepsAZeroMeasure(t *testing.T) {
	t.Parallel()
	data, err := json.Marshal(ScorecardMove{Name: "m", Task: "inference", Kind: "worse", Metric: "failure-rate", After: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"before":0`) || !strings.Contains(string(data), `"after":0.05`) {
		t.Fatalf("a move from zero marshals as %s, want its before kept", data)
	}
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
	// A published record that cannot be read costs the history, not the scorecard.
	got := read(func() (*ScorecardHistory, error) { return nil, errors.New("scorecard-x.json: unknown field") })
	if got.History == nil || got.History.Unavailable != "scorecard-x.json: unknown field" || len(got.History.Moves) != 0 || got.Rows == nil {
		t.Fatalf("an unreadable record answered %+v", got)
	}
}
