//go:build windows

package adaptiveparity_test

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
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
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	scorerDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	drafterDir := filepath.Join(roots.Models, "MiniCPM5-1B")
	for _, directory := range []string{scorerDir, drafterDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; composed artifact NOT verified", directory)
		}
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()

	scorerID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("tier0-chain/v1:model:"+scorerDir))
	if err != nil {
		t.Fatal(err)
	}
	drafterID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("tier0-chain/v1:model:"+drafterDir))
	if err != nil {
		t.Fatal(err)
	}
	chainProgram, _, err := composition.ChainPrograms(scorerID, drafterID)
	if err != nil {
		t.Fatal(err)
	}
	decomposition, err := modelartifact.NewComponentDecomposition(scorerID, "qwen2", "text",
		[]modelartifact.TensorFact{{Name: "blk.0.attn_q.weight", Shape: []uint64{896, 896}, Storage: "bf16"}})
	if err != nil {
		t.Fatal(err)
	}
	composed, err := composition.NewComposedModel(composition.ComposedModelDocument{
		Architecture: "qwen2",
		Recipe:       chainProgram.Definition().ID,
		Parents:      []artifact.ID{scorerID, drafterID},
		Components:   []artifact.ID{decomposition.ID},
	})
	if err != nil {
		t.Fatal(err)
	}
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

	recipeContent, err := chainProgram.Definition().ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	decompositionContent, err := decomposition.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "composed/inputs",
		Artifacts: []artifact.Descriptor{{ID: scorerID}, {ID: drafterID}},
		Contents:  []artifact.Content{recipeContent, decompositionContent},
	}); err != nil {
		t.Fatal(err)
	}
	batch, err := composed.Batch("composed/" + composed.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(ctx, composed.ID)
	if err != nil {
		t.Fatal(err)
	}
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
	result, err := composition.RunChainViability(store, composition.ChainConfig{
		ScorerDir: scorerDir, DrafterDir: drafterDir,
		Prefix: 64, Draft: 8, Target: 32,
		Windows: [][]int{tokens[:104]},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ChainRecipe != parsed.Recipe {
		t.Fatalf("execution recipe %s differs from the composed reference %s", result.ChainRecipe, parsed.Recipe)
	}
	t.Logf("composed artifact %s executed via recipe %s; lineage resolves %d inputs",
		composed.ID, result.ChainRecipe, len(parents))
}
