package agenttool

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestTransportEntryWaitHonorsDeadline pins the reopened finding: a
// wedged first invocation must not make a second call wait past its
// own bound -- the entry slot is acquired under the context, so the
// second call returns a typed timeout while the first still holds it.
func TestTransportEntryWaitHonorsDeadline(t *testing.T) {
	executor := NewExecutor()
	manual := inspectionManual(t, "probe.wedge", Transport{Kind: TransportBuiltin})
	release := make(chan struct{})
	entered := make(chan struct{})
	if err := executor.registerBuiltin("probe.wedge", func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
		close(entered)
		select {
		case <-release:
			return json.RawMessage(`{}`), nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}); err != nil {
		t.Fatal(err)
	}
	firstDone := make(chan error, 1)
	go func() {
		_, err := executor.Invoke(t.Context(), manual, json.RawMessage(`{"pattern":"x"}`))
		firstDone <- err
	}()
	<-entered
	bounded, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	started := time.Now()
	_, err := executor.Invoke(bounded, manual, json.RawMessage(`{"pattern":"x"}`))
	waited := time.Since(started)
	if err == nil || !strings.Contains(err.Error(), "waiting for entry") {
		t.Fatalf("second call = %v, want a typed entry-wait timeout", err)
	}
	if waited > 5*time.Second {
		t.Fatalf("second call waited %s; the deadline did not bound the entry wait", waited)
	}
	close(release)
	if err := <-firstDone; err != nil {
		t.Fatalf("first call failed after release: %v", err)
	}
}
