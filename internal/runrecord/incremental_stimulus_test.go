package runrecord

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func incrementalStimulusBoundary(t *testing.T, store *overgodb.Store, attempt uint32, payload string) AttemptStimulusBoundary {
	t.Helper()
	ctx := context.Background()
	content, err := AttemptArgumentContent([]byte(payload))
	if err != nil {
		t.Fatal(err)
	}
	size := content.Descriptor.Size
	head, _ := store.Head()
	selection, err := dataset.SelectInteractions(dataset.InteractionSelectionBounds{
		MaxTokens: size, MaxBytes: size, MaxDocuments: 1, MaxDepth: 1, MaxResults: 1,
	}, head, []dataset.InteractionSelectionSource{{
		Source: content.Descriptor.ID, CausalRoot: testutil.ArtifactID(t, artifact.KindEvidence, "op"),
		Tokens: size, Bytes: size, Documents: 1, Depth: 1,
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	boundary, err := PublishAttemptStimulus(ctx, store, AttemptStimulusBoundary{
		Operation: testutil.ArtifactID(t, artifact.KindEvidence, "op"), Attempt: attempt,
		Manual: testutil.ArtifactID(t, artifact.KindRecipe, "manual"), Selection: selection,
	}, []artifact.Content{content})
	if err != nil {
		t.Fatal(err)
	}
	return boundary
}

// TestIncrementalContextPreservesEvidence pins the boundary-layer proof: a
// cursor delta between two admitted stimulus boundaries verifies only when
// recomputation reproduces it exactly, so a runtime receiving the delta
// still observes every admitted stimulus; unidentified or identical
// boundaries refuse.
func TestIncrementalContextPreservesEvidence(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "incremental/seed",
		Artifacts: []artifact.Descriptor{
			{ID: testutil.ArtifactID(t, artifact.KindEvidence, "seed"), Size: 1},
			{ID: testutil.ArtifactID(t, artifact.KindEvidence, "op")},
			{ID: testutil.ArtifactID(t, artifact.KindRecipe, "manual")},
		},
	}); err != nil {
		t.Fatal(err)
	}
	held := incrementalStimulusBoundary(t, store, 1, `{"turn":1}`)
	current := incrementalStimulusBoundary(t, store, 2, `{"turn":2}`)
	delta, err := dataset.SelectIncremental(held.Selection, current.Selection)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyIncrementalStimulus(held, current, delta); err != nil {
		t.Fatal(err)
	}
	if len(delta.Fresh) != 1 || len(delta.Reused) != 0 {
		t.Fatalf("distinct turns = %+v", delta)
	}

	tampered := delta
	tampered.Fresh = nil
	if err := VerifyIncrementalStimulus(held, current, tampered); err == nil ||
		!strings.Contains(err.Error(), "differs from its admitted boundaries") {
		t.Fatalf("omitted stimulus verified: %v", err)
	}
	if err := VerifyIncrementalStimulus(held, held, delta); err == nil ||
		!strings.Contains(err.Error(), "distinct boundaries") {
		t.Fatalf("identical boundaries verified: %v", err)
	}
	if err := VerifyIncrementalStimulus(AttemptStimulusBoundary{}, current, delta); err == nil ||
		!strings.Contains(err.Error(), "identified boundaries") {
		t.Fatalf("unidentified boundary verified: %v", err)
	}
}
