package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// TestAReplayProbesTheWorkersSettingsFirst holds a replay of the local worker to one request at the worker's settings
// before any task (fail-early-runs/model-run-probe): the six-task thinking run of 2026-11-10 spent 23 minutes on a first
// task whose every reply came back empty. A route that answers an empty reply cut at its bound refuses the replay after
// that one request, the refusal naming why the reply ended; a route that answers a tool call lets it go on to its steps,
// which this tree of no commits does not hold.
func TestAReplayProbesTheWorkersSettingsFirst(t *testing.T) {
	t.Parallel()
	route := func(reply string) (*httptest.Server, *atomic.Int32) {
		var asked atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			asked.Add(1)
			_, _ = io.WriteString(writer, reply)
		}))
		t.Cleanup(server.Close)
		return server, &asked
	}
	empty, asked := route(`{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"length"}]}`)
	err := replaySteps(t.Context(), t.TempDir(), "2026-11-01", 1, localReplayWorker(empty.URL, workerModel{name: "fixture", thinking: true}, io.Discard), io.Discard)
	if err == nil || !strings.Contains(err.Error(), `"length"`) || !strings.Contains(err.Error(), "no task is started") || asked.Load() != 1 {
		t.Errorf("a route answering an empty reply cut at its bound gave %v after %d requests; want the replay refused after one, naming why the reply ended", err, asked.Load())
	}
	calling, _ := route(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"1","type":"function","function":{"name":"show_step","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
	err = replaySteps(t.Context(), t.TempDir(), "2026-11-01", 1, localReplayWorker(calling.URL, workerModel{name: "fixture"}, io.Discard), io.Discard)
	if err != nil && strings.Contains(err.Error(), "no task is started") {
		t.Errorf("a route answering a tool call refused the replay: %v", err)
	}
}
