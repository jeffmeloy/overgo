//go:build windows

package adaptiveparity_test

import (
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modelartifact"
	"overgo/internal/modeltest"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/trainingworkflow"
)

// TestComposedArtifactExecutesAndResolvesLineage pins the compose-model
// -artifact row: a composed model is ONE content-addressed artifact
// assembling parents, executable recipe, classified components and
// architecture vocabulary; committing it binds lineage to every input so
// provenance resolves as a store graph query; and the artifact EXECUTES --
// the referenced chain recipe runs the real model pair through the generic
// workflow runtime and reproduces the composed recipe identity.
func TestComposedArtifactExecutesAndResolvesLineage(t *testing.T) {
	cudatest.Require(t)
	scorerDir := modeltest.Directory(t, "Qwen2.5-0.5B")
	drafterDir := modeltest.Directory(t, "MiniCPM5-1B")
	for _, directory := range []string{scorerDir, drafterDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; composed artifact NOT verified", directory)
		}
	}
	store := overgodbtest.Open(t, t.TempDir())
	ctx := t.Context()

	scorerID := testcheck.Must(artifact.IdentifyBytes(artifact.KindModel, []byte("tier0-chain/v1:model:"+scorerDir)))(t)
	drafterID := testcheck.Must(artifact.IdentifyBytes(artifact.KindModel, []byte("tier0-chain/v1:model:"+drafterDir)))(t)
	chainProgram, _, err := trainingworkflow.ChainPrograms(scorerID, drafterID)
	testcheck.NoError(t, err)
	decomposition := testcheck.Must(modelartifact.NewComponentDecomposition(scorerID, "qwen2", "text",
		[]modelartifact.TensorFact{{Name: "blk.0.attn_q.weight", Shape: []uint64{896, 896}, Storage: "bf16"}}))(t)
	composed := testcheck.Must(composition.NewComposedModel(composition.ComposedModelDocument{
		Architecture: "qwen2",
		Recipe:       chainProgram.Definition().ID,
		Parents:      []artifact.ID{scorerID, drafterID},
		Components:   []artifact.ID{decomposition.ID},
	}))(t)
	replay, err := composition.NewComposedModel(composition.ComposedModelDocument{
		Architecture: "qwen2",
		Recipe:       chainProgram.Definition().ID,
		Parents:      []artifact.ID{scorerID, drafterID},
		Components:   []artifact.ID{decomposition.ID},
	})
	if err != nil || replay.ID != composed.ID {
		t.Fatalf("composed identity not deterministic: (%v, %v)", replay.ID, err)
	}
	if _, err := composition.NewComposedModel(composition.ComposedModelDocument{
		Architecture: "not-a-registered-family",
		Recipe:       chainProgram.Definition().ID,
		Parents:      []artifact.ID{scorerID},
	}); err == nil {
		t.Fatal("unregistered architecture accepted")
	}

	recipeContent := testcheck.Must(chainProgram.Definition().ArtifactContent())(t)
	decompositionContent := testcheck.Must(decomposition.Content())(t)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "composed/inputs",
		Artifacts: []artifact.Descriptor{{ID: scorerID}, {ID: drafterID}},
		Contents:  []artifact.Content{recipeContent, decompositionContent},
	}); err != nil {
		t.Fatal(err)
	}
	batch := testcheck.Must(composed.Batch("composed/" + composed.ID.String()))(t)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	parents := testcheck.Must(store.Parents(ctx, composed.ID))(t)
	resolved := map[artifact.ID]bool{}
	for _, parent := range parents {
		resolved[parent.Parent] = true
	}
	for _, input := range []artifact.ID{chainProgram.Definition().ID, scorerID, drafterID, decomposition.ID} {
		if !resolved[input] {
			t.Fatalf("lineage does not resolve input %s", input)
		}
	}
	stored, ok, err := artifact.ReadContent(ctx, store, composed.ID)
	if err != nil || !ok {
		t.Fatalf("composed artifact not durable: (%v, %v)", ok, err)
	}
	parsed, err := composition.ParseComposedModel(stored.Data)
	if err != nil || parsed.ID != composed.ID || parsed.Recipe != chainProgram.Definition().ID {
		t.Fatalf("roundtrip = (%+v, %v)", parsed, err)
	}

	// Execution: the composed artifact's recipe runs the REAL model pair
	// through the generic runtime; the run's chain recipe identity must be
	// exactly the composed artifact's recipe reference.
	tokens := qwenAdaptiveProfileTokens
	result := testcheck.Must(trainingworkflow.RunChainViability(store, trainingworkflow.ChainConfig{
		ScorerDir: scorerDir, DrafterDir: drafterDir,
		Prefix: 64, Draft: 8, Target: 32,
		Windows: [][]int{tokens[:104]},
	}))(t)
	if result.ChainRecipe != parsed.Recipe {
		t.Fatalf("execution recipe %s differs from the composed reference %s", result.ChainRecipe, parsed.Recipe)
	}
	t.Logf("composed artifact %s executed via recipe %s; lineage resolves %d inputs",
		composed.ID, result.ChainRecipe, len(parents))
}
