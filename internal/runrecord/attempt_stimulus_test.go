package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestAttemptStimulusBoundary(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	manual := testutil.ArtifactID(t, artifact.KindEvidence, "stimulus-manual")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "stimulus/authority", Artifacts: []artifact.Descriptor{{ID: manual}}}); err != nil {
		t.Fatal(err)
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "stimulus-operation")
	arguments, err := AttemptArgumentContent([]byte(`{"query":"before"}`))
	if err != nil {
		t.Fatal(err)
	}
	head, _ := store.Head()
	selection, err := dataset.SelectInteractions(dataset.InteractionSelectionBounds{
		MaxTokens: arguments.Descriptor.Size, MaxBytes: arguments.Descriptor.Size,
		MaxDocuments: 1, MaxDepth: 1, MaxResults: 1,
	}, head, []dataset.InteractionSelectionSource{{
		Source: arguments.Descriptor.ID, CausalRoot: operation, Tokens: arguments.Descriptor.Size,
		Bytes: arguments.Descriptor.Size, Documents: 1, Depth: 1,
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	boundary, err := PublishAttemptStimulus(ctx, store, AttemptStimulusBoundary{
		Operation: operation, Attempt: 1, Manual: manual, Selection: selection,
	}, []artifact.Content{arguments})
	if err != nil {
		t.Fatal(err)
	}
	late := testutil.ArtifactID(t, artifact.KindEvidence, "late-stimulus")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "stimulus/late", Artifacts: []artifact.Descriptor{{ID: late}}}); err != nil {
		t.Fatal(err)
	}
	resolved, found, err := ResolveAttemptStimulus(ctx, store, operation, 1)
	if err != nil || !found || resolved.ID != boundary.ID || resolved.Selection.Head != head ||
		len(resolved.Selection.Sources) != 1 || resolved.Selection.Sources[0].Source != arguments.Descriptor.ID {
		t.Fatalf("resolved boundary = (%+v, %t, %v)", resolved, found, err)
	}
	changedSelection, err := dataset.SelectInteractions(selection.Bounds, head, []dataset.InteractionSelectionSource{{
		Source: late, CausalRoot: operation, Tokens: arguments.Descriptor.Size,
		Bytes: arguments.Descriptor.Size, Documents: 1, Depth: 1,
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	changed := boundary
	changed.Selection = changedSelection
	if _, err := PublishAttemptStimulus(ctx, store, changed, nil); err == nil {
		t.Fatal("late stimulus rewrote the admitted boundary")
	}
}
