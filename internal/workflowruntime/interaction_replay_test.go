// Package workflowruntime tests artifact-owned replay without runtime callbacks.
package workflowruntime

import (
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type interactionReplayAdapter struct {
	trace runrecord.InteractionTrace
}

func (adapter interactionReplayAdapter) replay(request artifact.ID) ([]runrecord.InteractionTraceEvent, error) {
	if request != adapter.trace.Request {
		return nil, errors.New("workflow replay: request identity differs")
	}
	return append([]runrecord.InteractionTraceEvent(nil), adapter.trace.Events...), nil
}

func TestInteractionReplayExact(t *testing.T) {
	trace := replayTraceFixture(t)
	events, err := (interactionReplayAdapter{trace: trace}).replay(trace.Request)
	if err != nil || len(events) != len(trace.Events) || events[0].Message.Content != trace.Events[0].Message.Content {
		t.Fatalf("replay = (%+v, %v)", events, err)
	}
}

func TestReplayRejectsRequestDrift(t *testing.T) {
	trace := replayTraceFixture(t)
	drift := testutil.ArtifactID(t, artifact.KindEvidence, "drifted request")
	if _, err := (interactionReplayAdapter{trace: trace}).replay(drift); err == nil {
		t.Fatal("request drift was replayed")
	}
}

func replayTraceFixture(t *testing.T) runrecord.InteractionTrace {
	t.Helper()
	trace, err := runrecord.NewInteractionTrace(runrecord.Interaction{
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "replay recipe"),
		Model:  testutil.ArtifactID(t, artifact.KindModel, "replay model"),
	}, testutil.ArtifactID(t, artifact.KindEvidence, "replay request"), []runrecord.InteractionMessage{
		{Role: "assistant", Content: "answer"},
	}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	return trace
}
