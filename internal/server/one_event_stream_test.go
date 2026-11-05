package server

import (
	"testing"

	"overgo/internal/hfhub"
	"overgo/internal/testcheck"
)

// TestOneWorkbenchEventStream: hub downloads ride the one runtime stream --
// every job at connect, then again only when a job reads differently -- in
// place of a once-a-second poll.
func TestOneWorkbenchEventStream(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()

	// A transfer publishes once per file and whole percent, not per chunk,
	// and what it publishes is the progress it just recorded.
	var published []downloadJob
	registry := newDownloadRegistry(1, 1, func(_ string, value any) { published = value.([]downloadJob) })
	job := &downloadJob{ID: 1, State: downloadStateRunning}
	registry.jobs = map[uint64]*downloadJob{job.ID: job}
	woke := func(progress hfhub.Progress) bool {
		published = nil
		registry.observe(job, progress)
		if published != nil && (published[0].File != progress.Path || published[0].Received != progress.Received) {
			t.Errorf("progress %+v published as %+v", progress, published[0])
		}
		return published != nil
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

	// The stream stops before the handler closes.
	probe := openRuntimeServingProbe(t, handler)
	defer probe.stop()
	for _, event := range []string{"runtime.sessions", "runtime.activity", "operation.snapshot"} {
		probe.next(t, event, nil)
		probe.release(t)
	}
	var jobs []downloadJob
	if probe.next(t, "hub.downloads", &jobs); len(jobs) != 0 {
		t.Fatalf("a fresh server lists downloads %+v", jobs)
	}
	probe.release(t)
	probe.next(t, "resync", nil)
	probe.release(t)
	admitted := &downloadJob{ID: handler.downloads.next.Add(1), Repository: "acme/tiny", State: downloadStateRunning}
	testcheck.NoError(t, handler.downloads.admit(admitted, func() {}))
	defer handler.downloads.transfers.Done()
	if probe.next(t, "hub.downloads", &jobs); len(jobs) != 1 || jobs[0].Repository != "acme/tiny" || jobs[0].State != downloadStateRunning {
		t.Fatalf("an admitted download reached the stream as %+v", jobs)
	}
	probe.release(t)
	handler.downloads.cancel(admitted.ID)
	if probe.next(t, "hub.downloads", &jobs); len(jobs) != 1 || jobs[0].State != downloadStateCancelled {
		t.Fatalf("a cancelled download reached the stream as %+v", jobs)
	}
}
