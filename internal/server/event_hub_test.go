package server

import (
	"context"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestEventHubFansOutOnce holds the workbench's events to one fan-out: the
// operation manager notifies one hub, every stream sees each transition in
// order through to the last, the sessions those transitions changed are
// computed at most once per transition whatever the number of streams and
// reach every stream after the last one, and a stream a queue behind is told
// to resync instead of silently losing a transition.
func TestEventHubFansOutOnce(t *testing.T) {
	t.Parallel()
	hub := newEventHub()
	defer hub.close()
	var sessions atomic.Int64
	hub.pump(func() runtimeSessionsResponse { sessions.Add(1); return runtimeSessionsResponse{} })
	manager, err := operation.NewManager(8)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	var last atomic.Uint64
	manager.Notify(func(event operation.Event) { last.Store(event.Sequence); hub.observeOperation(event) })
	const streams = 3
	queues := make([]<-chan hubEvent, streams)
	for index := range queues {
		events, unsubscribe, err := hub.subscribe(64)
		if err != nil {
			t.Fatal(err)
		}
		defer unsubscribe()
		queues[index] = events
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "event hub recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "event hub run")
	id, err := manager.Submit(t.Context(), operation.Request{Task: recipe.TaskGeneration, Recipe: recipeID},
		func(_ context.Context, reporter operation.Reporter) (operation.Completion, error) {
			reporter.Metric(operation.Metric{Name: "loss", Value: 1})
			reporter.Publishing()
			return operation.Completion{Run: runID}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Wait(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	// Status takes the manager's lock, so the run's last transition has been published.
	if status, _ := manager.Status(id); status.State != operation.StateCompleted {
		t.Fatalf("operation ended %s", status.State)
	}
	final := last.Load()
	next := func(index int, events <-chan hubEvent) hubEvent {
		t.Helper()
		select {
		case event := <-events:
			return event
		case <-t.Context().Done():
			t.Fatalf("stream %d: %v", index, t.Context().Err())
			return hubEvent{}
		}
	}
	for index, events := range queues {
		// Each stream is told that the store gained the records a settled
		// run writes; a settled state's later transitions may tell it again.
		var sequence uint64
		records := 0
		for sequence != final {
			event := next(index, events)
			switch event.name {
			case "runtime.sessions":
			case hubRecordsChanged:
				records++
			case "operation":
				transition := event.value.(operation.Event)
				if transition.Sequence != sequence+1 {
					t.Fatalf("stream %d: transition %d after %d", index, transition.Sequence, sequence)
				}
				sequence = transition.Sequence
			default:
				t.Fatalf("stream %d: unexpected %q", index, event.name)
			}
		}
		for event := next(index, events); event.name != "runtime.sessions"; event = next(index, events) {
			if event.name != hubRecordsChanged {
				t.Fatalf("stream %d: %q after the last transition", index, event.name)
			}
			records++
		}
		if records == 0 {
			t.Fatalf("stream %d: never told that a settled run wrote records", index)
		}
	}
	if computed := sessions.Load(); computed < 1 || computed > int64(final) {
		t.Errorf("%d transitions to %d streams computed %d sessions snapshots", final, streams, computed)
	}

	// A stream that stops reading holds one resync, not a stale backlog.
	slow, unsubscribe, err := hub.subscribe(64)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribe()
	for range cap(slow) + 1 {
		hub.publish("hub.downloads", []DownloadJob{})
	}
	if event := <-slow; len(slow) != 0 || event.name != hubResync {
		t.Fatalf("an overflowed stream holds %q and %d more", event.name, len(slow))
	}
}
