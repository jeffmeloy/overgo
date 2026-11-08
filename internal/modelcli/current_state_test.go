package modelcli

import (
	"path/filepath"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testcheck"
)

// TestAJudgeWithStaleEvidenceIsVerifiedBeforeItJudges holds the state a judged run reads of its judge before opening it
// (cmd/evaluate holds the run's side) to the one an open would note, read without loading the weights: a model whose
// active recipe this build compiles reads verified, and one the store never verified reads not verified.
func TestAJudgeWithStaleEvidenceIsVerifiedBeforeItJudges(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	judge, unverified := filepath.Join(root, "judge.gguf"), filepath.Join(root, "unverified.gguf")
	writeServingFixture(t, judge, 0)
	writeServingFixture(t, unverified, 300)
	ctx := t.Context()
	store := overgodbtest.Create(t, filepath.Join(root, "store"))
	testcheck.NoError(t, servingtest.PublishActiveGGUFWithPolicy(ctx, store, judge, recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference))
	if state := testcheck.Must(CurrentState(ctx, store, judge, recipe.TaskInference))(t); state.Verification != modelrecipe.StateVerified {
		t.Fatalf("an activated model reads %+v, want verified", state)
	}
	if state := testcheck.Must(CurrentState(ctx, store, unverified, recipe.TaskInference))(t); state.Verification != modelrecipe.StateNotVerified {
		t.Fatalf("a model nothing verified reads %+v, want not verified", state)
	}
}
