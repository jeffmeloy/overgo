package runrecord

import (
	"sync"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/invocation"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestLateStimulusReconciliation(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	manual := testutil.ArtifactID(t, artifact.KindRecipe, "followup-manual")
	ceiling := testutil.ArtifactID(t, artifact.KindRecipe, "followup-ceiling")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "followup/authority", Artifacts: []artifact.Descriptor{{ID: manual}, {ID: ceiling}}}); err != nil {
		t.Fatal(err)
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "followup-operation")
	arguments := testcheck.Must(AttemptArgumentContent([]byte(`{"initial":true}`)))(t)
	effect := testcheck.Must(invocation.NewEffect(invocation.Effect{
		Manual: manual, Arguments: arguments.Descriptor.ID, Class: invocation.ClassInspection, Known: true,
	}))(t)
	effectContent := testcheck.Must(effect.Content())(t)
	head, _ := store.Head()
	selection := testcheck.Must(dataset.SelectInteractions(dataset.InteractionSelectionBounds{
		MaxTokens: arguments.Descriptor.Size, MaxBytes: arguments.Descriptor.Size,
		MaxDocuments: 1, MaxDepth: 1, MaxResults: 1,
	}, head, []dataset.InteractionSelectionSource{{
		Source: arguments.Descriptor.ID, CausalRoot: operation, Tokens: arguments.Descriptor.Size,
		Bytes: arguments.Descriptor.Size, Documents: 1, Depth: 1,
	}}, nil))(t)
	boundary := testcheck.Must(PublishAttemptStimulus(ctx, store, AttemptStimulusBoundary{
		Operation: operation, Attempt: 1, Manual: manual, Class: invocation.ClassInspection,
		Arguments: arguments.Descriptor.ID, Effect: effect.ID, Ceiling: ceiling, Selection: selection,
	}, []artifact.Content{arguments, effectContent}))(t)
	late := []artifact.ID{
		testutil.ArtifactID(t, artifact.KindEvidence, "late-a"),
		testutil.ArtifactID(t, artifact.KindEvidence, "late-b"),
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "followup/late", Artifacts: []artifact.Descriptor{{ID: late[0]}, {ID: late[1]}}}); err != nil {
		t.Fatal(err)
	}
	var won atomic.Uint32
	var wait sync.WaitGroup
	results := make(chan StimulusFollowup, 8)
	errors := make(chan error, 8)
	for range cap(results) {
		wait.Go(func() {
			admission, admitted, reconcileErr := (StimulusFollowupAuthority{Repository: store}).Admit(ctx, boundary.ID, []artifact.ID{late[1], late[0], late[1]})
			if reconcileErr != nil {
				errors <- reconcileErr
				return
			}
			if admitted {
				won.Add(1)
			}
			results <- admission
		})
	}
	wait.Wait()
	close(results)
	close(errors)
	for err := range errors {
		t.Fatal(err)
	}
	var identity artifact.ID
	for result := range results {
		if !identity.Valid() {
			identity = result.ID
		}
		if result.ID != identity || result.Boundary != boundary.ID || len(result.Sources) != len(late) ||
			result.Causal.Trigger != TriggerFollowup || result.Causal.FollowupOf != boundary.ID {
			t.Fatalf("follow-up differs: %+v", result)
		}
	}
	if won.Load() != 1 {
		t.Fatalf("follow-up winners = %d, want 1", won.Load())
	}
}
