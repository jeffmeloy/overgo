package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/invocation"
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
	manual := testutil.ArtifactID(t, artifact.KindRecipe, "stimulus-manual")
	ceiling := testutil.ArtifactID(t, artifact.KindRecipe, "stimulus-ceiling")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "stimulus/authority", Artifacts: []artifact.Descriptor{{ID: manual}, {ID: ceiling}}}); err != nil {
		t.Fatal(err)
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "stimulus-operation")
	arguments, err := AttemptArgumentContent([]byte(`{"query":"before"}`))
	if err != nil {
		t.Fatal(err)
	}
	effect, err := invocation.NewEffect(invocation.Effect{
		Manual: manual, Arguments: arguments.Descriptor.ID, Class: invocation.ClassInspection, Known: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	effectContent, err := effect.Content()
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
		Operation: operation, Attempt: 1, Manual: manual, Class: invocation.ClassInspection,
		Arguments: arguments.Descriptor.ID, Effect: effect.ID, Ceiling: ceiling, Selection: selection,
	}, []artifact.Content{arguments, effectContent})
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
