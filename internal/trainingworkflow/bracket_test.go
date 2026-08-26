package trainingworkflow

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestTrainingBracket pins the bracket contract: deltas exist only for
// suite metrics present on both sides plus the decode rate when both
// benchmarks stand, and the published bracket parses back intact with
// its session lineage.
func TestTrainingBracket(t *testing.T) {
	pre := BracketSlice{
		Benchmark: &evaluation.BenchmarkSummary{Tier: "capability-measured", DecodeTokensPerSecond: 400},
		Evals: []evaluation.EvalSummary{
			{Suite: "store/mmlu", Metrics: map[string]float64{"accuracy": 0.40}},
			{Suite: "store/musr", Metrics: map[string]float64{"accuracy": 0.30}},
		},
	}
	post := BracketSlice{
		Benchmark: &evaluation.BenchmarkSummary{Tier: "capability-measured", DecodeTokensPerSecond: 410},
		Evals: []evaluation.EvalSummary{
			{Suite: "store/mmlu", Metrics: map[string]float64{"accuracy": 0.46}},
			{Suite: "store/bbh", Metrics: map[string]float64{"accuracy": 0.50}},
		},
	}
	deltas := bracketDeltas(pre, post)
	if len(deltas) != 2 {
		t.Fatalf("deltas = %+v, want decode rate and the shared mmlu metric only", deltas)
	}
	if deltas[0].Suite != "benchmark" || deltas[0].Delta != 10 {
		t.Fatalf("benchmark delta = %+v", deltas[0])
	}
	if deltas[1].Suite != "store/mmlu" || deltas[1].Pre != 0.40 || deltas[1].Post != 0.46 {
		t.Fatalf("mmlu delta = %+v", deltas[1])
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	session := testutil.ArtifactID(t, artifact.KindEvidence, "bracket-session")
	model := testutil.ArtifactID(t, artifact.KindModel, "bracket-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "bracket-recipe")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/bracket/parents",
		Artifacts: []artifact.Descriptor{{ID: session}, {ID: model}},
	}); err != nil {
		t.Fatal(err)
	}
	id, err := publishTrainingBracket(ctx, store, session, model, recipeID, pre, post)
	if err != nil {
		t.Fatal(err)
	}
	bracket, found, err := trainingBracketCodec.Read(ctx, store, id)
	if err != nil || !found {
		t.Fatalf("bracket read = (%t, %v)", found, err)
	}
	if bracket.Session != session || bracket.Model != model || len(bracket.Deltas) != 2 ||
		bracket.Pre.Benchmark.DecodeTokensPerSecond != 400 {
		t.Fatalf("bracket = %+v", bracket)
	}
	parents, err := store.Parents(ctx, id)
	if err != nil || len(parents) < 2 {
		t.Fatalf("bracket lineage = (%+v, %v), want session and model parents", parents, err)
	}
}
