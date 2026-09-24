package server

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/hfhub"
)

// TestOneWorkbenchEventStream holds the workbench to one live connection:
// the automation, peer and agent tab streams are gone (each sent its
// inventory once, then only the operations the runtime stream carries), and
// hub downloads ride the runtime stream -- every job at connect, then again
// only when a job reads differently -- in place of a once-a-second poll.
func TestOneWorkbenchEventStream(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	for _, path := range []string{"/automations/stream", "/peers/stream", "/agents/stream"} {
		if response := serveTestRequest(handler, http.MethodGet, path, ""); response.Code != http.StatusNotFound {
			t.Errorf("retired tab stream %s answered %d: %s", path, response.Code, response.Body)
		}
	}

	// A transfer wakes watchers once per file and whole percent, not per chunk.
	registry := newDownloadRegistry(1, 1)
	job := &DownloadJob{ID: 1, State: downloadStateRunning}
	woke := func(progress hfhub.Progress) bool {
		_, changed := registry.watch()
		registry.observe(job, progress)
		select {
		case <-changed:
			return true
		default:
			return false
		}
	}
	for _, step := range []struct {
		progress hfhub.Progress
		wakes    bool
	}{
		{hfhub.Progress{Path: "model.gguf", Received: 0, Total: 1000}, true},
		{hfhub.Progress{Path: "model.gguf", Received: 9, Total: 1000}, false},
		{hfhub.Progress{Path: "model.gguf", Received: 10, Total: 1000}, true},
		{hfhub.Progress{Path: "config.json", Received: 10, Total: 1000}, true},
	} {
		if got := woke(step.progress); got != step.wakes {
			t.Errorf("progress %+v woke watchers = %v, want %v", step.progress, got, step.wakes)
		}
	}

	server := httptest.NewServer(handler)
	defer server.Close()
	// Closing the body ends the stream.
	request, err := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL+"/runtime/activity/stream", nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := server.Client().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	lines := bufio.NewScanner(response.Body)
	nextDownloads := func() []DownloadJob {
		t.Helper()
		event := ""
		for lines.Scan() {
			line := lines.Text()
			if name, found := strings.CutPrefix(line, "event: "); found {
				event = name
			}
			if data, found := strings.CutPrefix(line, "data: "); found && event == "hub.downloads" {
				var jobs []DownloadJob
				if err := json.Unmarshal([]byte(data), &jobs); err != nil {
					t.Fatal(err)
				}
				return jobs
			}
		}
		t.Fatalf("the runtime stream ended before hub.downloads: %v", lines.Err())
		return nil
	}
	if jobs := nextDownloads(); len(jobs) != 0 {
		t.Fatalf("a fresh server lists downloads %+v", jobs)
	}
	admitted := &DownloadJob{ID: handler.downloads.next.Add(1), Repository: "acme/tiny", State: downloadStateRunning}
	if err := handler.downloads.admit(admitted, func() {}); err != nil {
		t.Fatal(err)
	}
	defer handler.downloads.transfers.Done()
	if jobs := nextDownloads(); len(jobs) != 1 || jobs[0].Repository != "acme/tiny" || jobs[0].State != downloadStateRunning {
		t.Fatalf("an admitted download reached the stream as %+v", jobs)
	}
	handler.downloads.cancel(admitted.ID)
	if jobs := nextDownloads(); len(jobs) != 1 || jobs[0].State != downloadStateCancelled {
		t.Fatalf("a cancelled download reached the stream as %+v", jobs)
	}
}
