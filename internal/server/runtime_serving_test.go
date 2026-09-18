package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Each flushed frame parks until the test releases it. Publication while parked
// exercises a slow reader without sleep, polling, or a machine-specific timeout.
type runtimeServingRecorder struct {
	*httptest.ResponseRecorder
	ctx     context.Context
	frames  chan string
	advance chan struct{}
	offset  int
}

func (recorder *runtimeServingRecorder) Flush() {
	recorder.ResponseRecorder.Flush()
	body := recorder.Body.String()
	frame := body[recorder.offset:]
	recorder.offset = len(body)
	select {
	case recorder.frames <- frame:
	case <-recorder.ctx.Done():
		return
	}
	select {
	case <-recorder.advance:
	case <-recorder.ctx.Done():
	}
}

type runtimeServingProbe struct {
	recorder *runtimeServingRecorder
	cancel   context.CancelCauseFunc
	done     chan struct{}
}

func openRuntimeServingProbe(t *testing.T, handler *Handler) *runtimeServingProbe {
	t.Helper()
	ctx, cancel := context.WithCancelCause(t.Context())
	probe := &runtimeServingProbe{
		recorder: &runtimeServingRecorder{
			ResponseRecorder: httptest.NewRecorder(), ctx: ctx,
			frames: make(chan string), advance: make(chan struct{}),
		},
		cancel: cancel, done: make(chan struct{}),
	}
	go func() {
		handler.ServeHTTP(probe.recorder, httptest.NewRequest(http.MethodGet, "/runtime/activity/stream", nil).WithContext(ctx))
		close(probe.done)
	}()
	t.Cleanup(probe.stop)
	return probe
}

func (probe *runtimeServingProbe) stop() {
	probe.cancel(context.Canceled)
	<-probe.done
}

func (probe *runtimeServingProbe) next(t *testing.T, want string, target any) {
	t.Helper()
	select {
	case frame := <-probe.recorder.frames:
		name, rest, ok := strings.Cut(frame, "\n")
		if !ok || name != "event: "+want {
			t.Fatalf("wanted %s, got %q", want, frame)
		}
		if target != nil {
			if err := json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(rest, "data: "))), target); err != nil {
				t.Fatal(err)
			}
		}
	case <-probe.done:
		t.Fatalf("stream closed before %s", want)
	case <-t.Context().Done():
		t.Fatal(t.Context().Err())
	}
}

func (probe *runtimeServingProbe) release(t *testing.T) {
	t.Helper()
	select {
	case probe.recorder.advance <- struct{}{}:
	case <-probe.done:
		t.Fatal("stream closed while releasing a frame")
	case <-t.Context().Done():
		t.Fatal(t.Context().Err())
	}
}

func servingStreamFixture(t *testing.T, limit int) (*Handler, *overgodb.Store) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	generator := responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"visible answer"}})
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "stream/model", Artifacts: []artifact.Descriptor{{ID: generator.ModelID()}},
	}); err != nil {
		t.Fatal(err)
	}
	handler := newTestHandlerForRepository(t, store, generator)
	handler.config.MaxStoredResponses = limit
	return handler, store
}

func servingStreamTurn(t *testing.T, handler *Handler) {
	t.Helper()
	before := handler.observationErrors.Load()
	response := serveTestRequest(handler, http.MethodPost, "/v1/completions", `{"prompt":"stream fixture","max_tokens":1}`)
	if response.Code != http.StatusOK {
		t.Fatalf("completion status=%d: %s", response.Code, response.Body.String())
	}
	if failures := handler.observationErrors.Load(); failures != before {
		t.Fatalf("turn failed to publish its observation: failures %d -> %d", before, failures)
	}
}

func servingStreamInitial(t *testing.T, probe *runtimeServingProbe) runtimeActivityResponse {
	t.Helper()
	probe.next(t, "runtime.sessions", nil)
	probe.release(t)
	var snapshot runtimeActivityResponse
	probe.next(t, "runtime.activity", &snapshot)
	probe.release(t)
	probe.next(t, "operation.snapshot", nil)
	return snapshot // the reader remains parked after the initial snapshot
}

func TestRuntimeStreamDeliversServingObservation(t *testing.T) {
	t.Run("durable_live_failure_and_reconnect", func(t *testing.T) {
		handler, store := servingStreamFixture(t, 2)
		probe := openRuntimeServingProbe(t, handler)
		initial := servingStreamInitial(t, probe)
		if initial.Count != 0 || initial.Cursor != 0 || initial.Limit != handler.config.MaxStoredResponses {
			t.Fatalf("initial snapshot = %+v", initial)
		}
		if id := handler.publishServing(t.Context(), runrecord.ServingObservation{}); id.Valid() {
			t.Fatal("invalid observation was published")
		}
		handler.servingEvents.mu.Lock()
		failedCursor := handler.servingEvents.cursor
		buffered := 0
		for channel := range handler.servingEvents.watches {
			buffered += len(channel)
		}
		handler.servingEvents.mu.Unlock()
		if failedCursor != initial.Cursor || buffered != 0 {
			t.Fatalf("failed publication emitted an event: cursor=%d buffered=%d", failedCursor, buffered)
		}
		servingStreamTurn(t, handler)
		probe.release(t)
		var event runtimeServingEvent
		probe.next(t, "runtime.serving", &event)
		if event.Cursor != initial.Cursor+1 || event.Activity == nil || !event.Activity.ID.Valid() || event.PublishFail != handler.observationErrors.Load() {
			t.Fatalf("live event = %+v", event)
		}
		retained, err := runrecord.RequireServingObservation(t.Context(), store, event.Activity.ID)
		if err != nil || retained.ID != event.Activity.ID || retained.Usage != event.Activity.Usage {
			t.Fatalf("event preceded durable publication: %+v, %v", retained, err)
		}
		probe.stop()
		handler.servingEvents.mu.Lock()
		watchers := len(handler.servingEvents.watches)
		handler.servingEvents.mu.Unlock()
		if watchers != 0 {
			t.Fatalf("cancelled stream retained %d subscriptions", watchers)
		}
		reconnected := openRuntimeServingProbe(t, handler)
		snapshot := servingStreamInitial(t, reconnected)
		if snapshot.Cursor != event.Cursor || snapshot.Count != 1 || snapshot.Activity[0].ID != event.Activity.ID {
			t.Fatalf("reconnect lost or duplicated a retained turn: %+v", snapshot)
		}
		servingStreamTurn(t, handler)
		reconnected.release(t)
		var next runtimeServingEvent
		reconnected.next(t, "runtime.serving", &next)
		if next.Cursor != event.Cursor+1 || next.Activity == nil || next.Activity.ID == event.Activity.ID {
			t.Fatalf("reconnected live event = %+v", next)
		}
		reconnected.stop()
		if err := handler.Close(); err != nil {
			t.Fatal(err)
		}
		restarted, err := New(handler.config, handler.generator)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = restarted.Close() })
		afterRestart := openRuntimeServingProbe(t, restarted)
		restored := servingStreamInitial(t, afterRestart)
		if restored.Cursor != 0 || restored.Count != 2 || restored.Activity[0].ID != next.Activity.ID {
			t.Fatalf("restart lost durable history or reused an old cursor: %+v", restored)
		}
		servingStreamTurn(t, restarted)
		afterRestart.release(t)
		var fresh runtimeServingEvent
		afterRestart.next(t, "runtime.serving", &fresh)
		if fresh.Cursor != 1 || fresh.Activity == nil || fresh.Activity.ID == next.Activity.ID {
			t.Fatalf("restarted live event = %+v", fresh)
		}
	})
	t.Run("slow_reader_resnapshot", func(t *testing.T) {
		handler, _ := servingStreamFixture(t, 1)
		probe := openRuntimeServingProbe(t, handler)
		servingStreamInitial(t, probe)
		turns := handler.config.MaxStoredResponses + 2
		for range turns {
			servingStreamTurn(t, handler)
		}
		probe.release(t)
		var snapshot runtimeActivityResponse
		probe.next(t, "runtime.activity", &snapshot)
		if snapshot.Cursor != uint64(turns) || snapshot.Count != handler.config.MaxStoredResponses || !snapshot.Truncated {
			t.Fatalf("overflow was silently dropped: %+v", snapshot)
		}
		servingStreamTurn(t, handler)
		probe.release(t)
		var event runtimeServingEvent
		probe.next(t, "runtime.serving", &event)
		if event.Cursor != snapshot.Cursor+1 || event.Activity == nil || event.Activity.ID == snapshot.Activity[0].ID {
			t.Fatalf("delivery after overflow = %+v", event)
		}
	})
	t.Run("snapshot_overlap_keeps_cursor", func(t *testing.T) {
		handler, _ := servingStreamFixture(t, 2)
		probe := openRuntimeServingProbe(t, handler)
		probe.next(t, "runtime.sessions", nil) // subscribe, then park before the activity snapshot
		turns := handler.config.MaxStoredResponses - 1
		for range turns {
			servingStreamTurn(t, handler)
		}
		probe.release(t)
		var snapshot runtimeActivityResponse
		probe.next(t, "runtime.activity", &snapshot)
		if snapshot.Cursor != uint64(turns) || snapshot.Count != turns || snapshot.Truncated {
			t.Fatalf("overlapped snapshot = %+v", snapshot)
		}
		probe.release(t)
		probe.next(t, "operation.snapshot", nil)
		// Both events fit the buffer. The first belongs to the already-captured
		// snapshot; only the new publication may be delivered as a live event.
		servingStreamTurn(t, handler)
		probe.release(t)
		var next runtimeServingEvent
		probe.next(t, "runtime.serving", &next)
		if next.Cursor != snapshot.Cursor+1 || next.Activity == nil || next.Activity.ID == snapshot.Activity[0].ID {
			t.Fatalf("old backlog reinserted an evicted turn: %+v", next)
		}
	})
}
