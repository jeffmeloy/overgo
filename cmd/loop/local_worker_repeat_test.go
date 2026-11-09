package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// TestTheLocalWorkerStopsRepeatingItself: a reading the local worker made before is not run again while nothing is
// edited, and three replies in a row of repeats alone end the attempt, so the step goes to the next attempt or worker
// in place of the replies left. Qwen3.8-27B searched the same four texts over 50 times on one step on 2026-11-09. After
// an edit, the same reading runs again, since the checkout changed.
func TestTheLocalWorkerStopsRepeatingItself(t *testing.T) {
	t.Parallel()
	search := `{"name": "search_files", "arguments": "{\"text\": \"needle\"}"}`
	edit := `{"name": "edit_file", "arguments": "{\"path\": \"a.go\", \"old\": \"x\", \"new\": \"y\"}"}`
	for _, example := range []struct {
		name     string
		replies  []string
		searches int
		stopped  bool
	}{
		{"repeating", []string{search, search, search, search, search}, 1, true},
		{"after an edit", []string{search, edit, search, ""}, 2, false},
	} {
		t.Run(example.name, func(t *testing.T) {
			var mu sync.Mutex
			asked := 0
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				mu.Lock()
				call := example.replies[min(asked, len(example.replies)-1)]
				asked++
				mu.Unlock()
				message := `{"role": "assistant", "content": "done"}`
				if call != "" {
					message = fmt.Sprintf(`{"role": "assistant", "content": "", "tool_calls": [{"id": "c%d", "type": "function", "function": %s}]}`, asked, call)
				}
				fmt.Fprintf(writer, `{"choices": [{"message": %s}]}`, message)
			}))
			defer server.Close()
			searches := 0
			tools := []workerTool{
				{"search_files", "", []string{"text"}, func(map[string]string) (string, error) { searches++; return "a.go:1:needle", nil }},
				{"edit_file", "", []string{"path", "old", "new"}, func(map[string]string) (string, error) { return "edited a.go", nil }},
			}
			var log strings.Builder
			work, err := workLocally(t.Context(), server.URL, "model", "the step", tools, &log)
			if searches != example.searches || (err != nil) != example.stopped {
				t.Fatalf("the search ran %d times and the attempt ended with %v after %d replies, want %d runs and a stop %t:\n%s",
					searches, err, work.Turns, example.searches, example.stopped, log.String())
			}
			if example.stopped && (work.Turns != 4 || !strings.Contains(err.Error(), "repeated")) {
				t.Fatalf("the attempt stopped after %d replies with %v, want after the fourth, saying it repeated itself", work.Turns, err)
			}
		})
	}
}
