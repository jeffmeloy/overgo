package agentloop

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestDurableInvocationDivergenceAndEviction pins: a durable session logs
// each executed step as a child attempt; opening the session again evicts
// the earlier invocation, which can neither replay nor execute; a resumed
// invocation whose request at a recorded step differs from the log is
// refused with a typed divergence and the divergence is recorded as a
// failure observation; a matching request replays from the log and the
// session continues with new steps.
func TestDurableInvocationDivergenceAndEviction(t *testing.T) {
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	first, err := coordinator.OpenDurableSession(ctx, "durable-session")
	if err != nil {
		t.Fatal(err)
	}
	result, err := coordinator.Propose(ctx, first, "probe.read", json.RawMessage(`{"step":1}`))
	if err != nil || !strings.Contains(string(result), "seen") || first.Steps != 1 {
		t.Fatalf("first step = %s steps=%d, %v", result, first.Steps, err)
	}
	recorded := first.Interaction
	if child, found, err := first.attempt.Lookup(ctx, store, runrecord.DurableChild, "1"); err != nil || !found || child.Result != recorded {
		t.Fatalf("child entry = %+v found=%t, %v", child, found, err)
	}

	second, err := coordinator.OpenDurableSession(ctx, "durable-session")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, first, "probe.read", json.RawMessage(`{"step":2}`)); !errors.Is(err, runrecord.ErrStaleInvocation) {
		t.Fatalf("evicted invocation executed a step: %v", err)
	}
	if _, err := coordinator.Propose(ctx, second, "probe.read", json.RawMessage(`{"step":9}`)); !errors.Is(err, ErrReplayDivergence) {
		t.Fatalf("divergent replay = %v", err)
	}
	observations := failureObservations(t, store)
	if len(observations) != 1 || !strings.Contains(observations[0].Message, "divergence") || !strings.Contains(observations[0].Message, `{"step":9}`) {
		t.Fatalf("divergence observations = %+v", observations)
	}

	third, err := coordinator.OpenDurableSession(ctx, "durable-session")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, second, "probe.read", json.RawMessage(`{"step":1}`)); !errors.Is(err, runrecord.ErrStaleInvocation) {
		t.Fatalf("superseded invocation replayed: %v", err)
	}
	replayed, err := coordinator.Propose(ctx, third, "probe.read", json.RawMessage(`{"step":1}`))
	if err != nil || string(replayed) != string(result) || third.Steps != 1 || third.Interaction != recorded {
		t.Fatalf("replayed step = %s steps=%d interaction=%s, %v", replayed, third.Steps, third.Interaction, err)
	}
	if _, err := coordinator.Propose(ctx, third, "probe.read", json.RawMessage(`{"step":2}`)); err != nil || third.Steps != 2 {
		t.Fatalf("new step after replay = steps=%d, %v", third.Steps, err)
	}
	if child, found, err := third.attempt.Lookup(ctx, store, runrecord.DurableChild, "2"); err != nil || !found || child.Invocation != 3 || child.Result != third.Interaction {
		t.Fatalf("second child entry = %+v found=%t, %v", child, found, err)
	}
	if len(failureObservations(t, store)) != 1 {
		t.Fatal("a matching replay recorded a divergence")
	}
}

// failureObservations: every committed failure observation.
func failureObservations(t *testing.T, store *overgodb.Store) []runrecord.FailureObservation {
	t.Helper()
	result, err := store.Query(t.Context(), overgodb.Query{
		Kind: artifact.KindEvidence, MediaType: runrecord.FailureObservationMediaType, Schema: runrecord.FailureObservationSchema,
		MaxResults: 16, Projection: overgodb.ProjectContentPresence,
	})
	if err != nil {
		t.Fatal(err)
	}
	var observations []runrecord.FailureObservation
	for _, content := range result.Contents {
		observation, err := runrecord.RequireFailureObservation(t.Context(), store, content.Artifact)
		if err != nil {
			t.Fatal(err)
		}
		observations = append(observations, observation)
	}
	return observations
}
