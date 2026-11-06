package runrecord

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestInteractionTerminalReason(t *testing.T) {
	path := t.TempDir()
	store := testcheck.Must(overgodb.Create(path))(t)
	defer func() { _ = store.Close() }()
	value := Interaction{Response: "limited", Node: interactionTestNode,
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "limited-recipe"),
		Model:  testutil.ArtifactID(t, artifact.KindModel, "limited-model")}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "limited-authority", Artifacts: []artifact.Descriptor{{ID: value.Recipe}}}); err != nil {
		t.Fatal(err)
	}
	prompt := []InteractionMessage{{Role: "user", Content: "question"}}
	reserved := testcheck.Must(PublishInteraction(t.Context(), store, value, prompt, OutcomeInconclusive))(t)
	value.TerminalReason = InteractionOutputLimit
	messages := append(prompt, InteractionMessage{Role: "assistant", Content: "partial answer"})
	for _, outcome := range []Outcome{"", OutcomeInconclusive, OutcomeCancelled, OutcomeFailed} {
		if _, err := PublishInteraction(t.Context(), store, value, messages, outcome); err == nil {
			t.Fatalf("limit accepted for %q", outcome)
		}
	}
	value.TerminalReason = "invented"
	if _, err := PublishInteraction(t.Context(), store, value, messages, OutcomeSucceeded); err == nil {
		t.Fatal("unknown terminal reason accepted")
	}
	retained, found, err := ResolveInteraction(t.Context(), store, value.Response)
	if err != nil || !found || retained.ID != reserved.ID {
		t.Fatal("refused publication changed the reserved interaction")
	}
	value.TerminalReason = InteractionOutputLimit
	final, err := PublishInteraction(t.Context(), store, value, messages, OutcomeSucceeded)
	if err != nil || final.ID == reserved.ID {
		t.Fatalf("terminal publication = %+v, %v", final, err)
	}
	testcheck.NoError(t, store.Close())
	store = testcheck.Must(overgodb.Create(path))(t)
	loaded, found, err := ResolveInteraction(t.Context(), store, value.Response)
	if err != nil || !found || loaded.ID != final.ID || loaded.TerminalReason != InteractionOutputLimit {
		t.Fatalf("reopened interaction = %+v, %v, %v", loaded, found, err)
	}
	repeated, err := PublishInteraction(t.Context(), store, value, messages, OutcomeSucceeded)
	if err != nil || repeated.ID != final.ID {
		t.Fatalf("idempotent repeat = %+v, %v", repeated, err)
	}
	value.TerminalReason = ""
	if _, err := PublishInteraction(t.Context(), store, value, messages, OutcomeSucceeded); err == nil {
		t.Fatal("late completion erased output-limit reason")
	}
	// Existing records omit the optional reason and retain their exact identity.
	legacy := testcheck.Must(interactionCodec.NewInitial(Interaction{Response: "legacy", Node: interactionTestNode, Recipe: value.Recipe, Model: value.Model, Message: final.Message, Trace: final.Trace}))(t)
	content := testcheck.Must(interactionCodec.Content(legacy))(t)
	if bytes.Contains(content.Data, []byte("terminal_reason")) {
		t.Fatal("optional reason changed legacy document bytes")
	}
	parsed, err := ParseInteraction(content.Data)
	if err != nil || parsed.ID != legacy.ID || parsed.TerminalReason != "" {
		t.Fatalf("legacy round trip = %+v, %v", parsed, err)
	}
	// The codec owns copying; callers cannot mutate a constructed trace by
	// changing the slices that were passed to its constructor.
	value.Tools = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "tool action")}
	decisions := []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "decision")}
	trace := testcheck.Must(NewInteractionTrace(value, final.Message, messages, decisions, OutcomeSucceeded))(t)
	value.Tools[0], decisions[0] = artifact.ID{}, artifact.ID{}
	if err := trace.ValidateIdentity(); err != nil {
		t.Fatalf("trace retained caller-owned slices: %v", err)
	}
}
