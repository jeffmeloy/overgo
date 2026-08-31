package agentloop

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

func TestAttemptStimulusBoundary(t *testing.T) {
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "stimulus-session"}
	arguments := json.RawMessage(`{"step":1}`)
	if _, err := coordinator.Propose(ctx, session, "probe.read", arguments); err != nil {
		t.Fatal(err)
	}
	operation, err := MutationReceiptOperation("stimulus-session-step-1")
	if err != nil {
		t.Fatal(err)
	}
	boundary, found, err := runrecord.ResolveAttemptStimulus(ctx, store, operation, 1)
	if err != nil || !found {
		t.Fatalf("attempt boundary = (%+v, %t, %v)", boundary, found, err)
	}
	interaction, found, err := runrecord.ResolveInteraction(ctx, store, "stimulus-session-step-1")
	if err != nil || !found || interaction.Stimulus != boundary.ID {
		t.Fatalf("interaction stimulus = (%+v, %t, %v)", interaction, found, err)
	}
	content, found, err := artifact.ReadContent(ctx, store, boundary.Selection.Sources[0].Source)
	if err != nil || !found || string(content.Data) != string(arguments) {
		t.Fatalf("admitted arguments = (%q, %t, %v)", content.Data, found, err)
	}
	current, _ := store.Head()
	if boundary.Selection.Head == current {
		t.Fatal("boundary silently advanced through execution commits")
	}
}
