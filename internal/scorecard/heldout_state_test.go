package scorecard

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAHeldOutReadingKeepsTheStateItRanUnder: a reading says what was known of its model, as an attempt does, and a record
// from before that keeps its identity. A candidate read before it activated reaches no row of the verified model, whose
// other readings stand, and a model nothing has verified keeps its own.
func TestAHeldOutReadingKeepsTheStateItRanUnder(t *testing.T) {
	t.Parallel()
	metrics := []runrecord.Metric{{Name: "accuracy", Value: 0.5, Direction: runrecord.DirectionMaximize}}
	plain, err := runrecord.NewEvaluation(testutil.ArtifactID(t, artifact.KindRecipe, "candidate"), testutil.ArtifactID(t, artifact.KindRun, "run"),
		testutil.ArtifactID(t, artifact.KindDataset, "suite"), metrics)
	testcheck.NoError(t, err)
	read, err := plain.UnderState(string(modelrecipe.StateNotVerified))
	testcheck.NoError(t, err)
	content, err := read.Content()
	testcheck.NoError(t, err)
	parsed, err := runrecord.ParseEvaluation(content.Data)
	if err != nil || parsed.Verification != string(modelrecipe.StateNotVerified) || parsed.ID == plain.ID {
		t.Fatalf("the reading came back as %+v (%v), want its state, under an identity of its own", parsed, err)
	}
	unchanged, err := plain.UnderState("")
	if err != nil || unchanged.ID != plain.ID {
		t.Fatalf("a record with no state moved from %s to %s (%v)", plain.ID, unchanged.ID, err)
	}
	readings := []evaluation.EvalSummary{
		{Suite: "before", Verification: string(modelrecipe.StateNotVerified)}, {Suite: "broken", Verification: string(modelrecipe.StateBroken)},
		{Suite: "after", Verification: string(modelrecipe.StateVerified)}, {Suite: "older"},
	}
	kept := earned(readings, modelrecipe.ModelState{Verification: modelrecipe.StateVerified})
	if len(kept) != 2 || kept[0].Suite != "after" || kept[1].Suite != "older" {
		t.Fatalf("a verified model's row keeps %+v", kept)
	}
	if kept := earned(readings, modelrecipe.ModelState{Verification: modelrecipe.StateNotVerified}); len(kept) != len(readings) {
		t.Fatalf("a model nothing verified keeps %d of its %d readings", len(kept), len(readings))
	}
}
