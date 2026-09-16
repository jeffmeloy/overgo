package agenttool

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
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
	// The first call holds the entry throughout, so the second call's wait
	// ends only with its caller: a cancelled caller is answered at once.
	ended := errors.New("the caller left the entry wait")
	bounded, cancel := context.WithCancelCause(t.Context())
	cancel(ended)
	_, err := executor.Invoke(bounded, manual, json.RawMessage(`{"pattern":"x"}`))
	if err == nil || !strings.Contains(err.Error(), "waiting for entry") || !errors.Is(err, ended) {
		t.Fatalf("second call = %v, want a typed entry-wait end with the caller's cause", err)
	}
	close(release)
	if err := <-firstDone; err != nil {
		t.Fatalf("first call failed after release: %v", err)
	}
}
