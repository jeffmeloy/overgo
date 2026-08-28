package modelrecipe

import (
	"context"
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestHighFanoutReadsAreBatched proves decision loading over a mixed
// evidence list resolves through one presence acquisition and one
// batched content read -- two acquisitions total: absent entries skip,
// and per-item opens do not scale with the list.
func TestHighFanoutReadsAreBatched(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	subject := testutil.ArtifactID(t, artifact.KindRecipe, "batched-subject")
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "batched-derivation")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "batched/authority",
		Artifacts: []artifact.Descriptor{{ID: subject}, {ID: derivation}},
	}); err != nil {
		t.Fatal(err)
	}
	ids := make([]artifact.ID, 0, 12)
	const decisions = 8
	for ordinal := range decisions {
		decision, err := recipe.NewDecision(
			subject, recipe.DecisionRefused, recipe.EvidenceParity, fmt.Sprintf("batched decision %d", ordinal),
			recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: derivation},
			[]artifact.ID{derivation},
		)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := decision.Batch(fmt.Sprintf("batched/decision/%d", ordinal))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, decision.ID)
	}
	ids = append(ids, testutil.ArtifactID(t, artifact.KindEvidence, "batched-absent"))
	counting := &repositorytest.CountingRepository{Repository: store}
	loaded, err := loadDecisions(ctx, counting, ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded) != decisions {
		t.Fatalf("loaded %d decisions, want %d", len(loaded), decisions)
	}
	if counting.Opens != 0 || counting.Batches != 2 {
		t.Fatalf("decision load used opens=%d batches=%d", counting.Opens, counting.Batches)
	}
}
