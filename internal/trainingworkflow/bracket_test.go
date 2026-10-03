package trainingworkflow

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestTrainingBracketMatchesExecutionContracts refuses cross-policy and unknown
// comparisons, and binds each accepted delta to the exact before/after records.
func TestTrainingBracketMatchesExecutionContracts(t *testing.T) {
	t.Parallel()
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "bracket-environment")
	resident := testutil.ArtifactID(t, artifact.KindProfile, "resident-comparison")
	isolated := testutil.ArtifactID(t, artifact.KindProfile, "isolated-comparison")
	summary := func(value float64, comparison artifact.ID, known bool, name string) evaluation.EvalSummary {
		t.Helper()
		data, err := json.Marshal(struct {
			evaluation.EvalSummary
			Comparison  artifact.ID `json:"comparison,omitzero"`
			Environment artifact.ID `json:"environment"`
			CodeCommit  string      `json:"code_commit"`
		}{evaluation.EvalSummary{Suite: "fixture", Prompting: "raw-completion", Record: testutil.ArtifactID(t, artifact.KindEvaluation, name),
			Metrics: map[string]float64{"accuracy": value}, Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize}, ExecutionKnown: known}, comparison, environment, "0123456789abcdef0123456789abcdef01234567"})
		var result evaluation.EvalSummary
		if err == nil {
			err = json.Unmarshal(data, &result)
		}
		if err != nil {
			t.Fatal(err)
		}
		result.Directions = map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize}
		return result
	}
	before := summary(0.4, resident, true, "before-resident")
	other := summary(0.8, isolated, true, "before-isolated")
	for _, scenario := range []struct {
		name    string
		post    evaluation.EvalSummary
		matched bool
	}{
		{"matching policy", summary(0.5, resident, true, "after-resident"), true},
		{"another environment", summary(0.5, testutil.ArtifactID(t, artifact.KindProfile, "other-environment-comparison"), true, "after-environment"), false},
		{"missing comparison", summary(0.5, artifact.ID{}, true, "after-missing"), false},
		{"unknown execution", summary(0.5, resident, false, "after-unknown"), false},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			deltas := bracketDeltas(BracketSlice{Evals: []evaluation.EvalSummary{before, other}}, BracketSlice{Evals: []evaluation.EvalSummary{scenario.post}})
			if !scenario.matched {
				if len(deltas) != 0 {
					t.Fatalf("incompatible delta = %+v", deltas)
				}
				return
			}
			if len(deltas) != 1 || deltas[0].Pre != 0.4 || deltas[0].Post != 0.5 {
				t.Fatalf("deltas = %+v, want the resident comparison", deltas)
			}
			data, err := json.Marshal(deltas[0])
			var evidence struct {
				Pre  artifact.ID `json:"pre_record"`
				Post artifact.ID `json:"post_record"`
			}
			if err == nil {
				err = json.Unmarshal(data, &evidence)
			}
			if err != nil || evidence.Pre != before.Record || evidence.Post != scenario.post.Record {
				t.Fatalf("delta evidence = %+v: %v", evidence, err)
			}
		})
	}
}

// TestTrainingBracket pins the bracket contract: deltas exist only for
// suite metrics present on both sides plus the decode rate when both
// benchmarks stand, and the published bracket parses back intact with
// its session lineage.
func TestTrainingBracket(t *testing.T) {
	t.Parallel()
	comparison := testutil.ArtifactID(t, artifact.KindProfile, "bracket-protocol-environment")
	pre := BracketSlice{
		Benchmark: &evaluation.BenchmarkSummary{Tier: "capability-measured", DecodeTokensPerSecond: 400},
		Evals: []evaluation.EvalSummary{
			{Suite: "store/mmlu", Metrics: map[string]float64{"accuracy": 0.40}, Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize}, ExecutionKnown: true, Comparison: comparison, Record: testutil.ArtifactID(t, artifact.KindEvaluation, "pre-mmlu")},
			{Suite: "store/musr", Metrics: map[string]float64{"accuracy": 0.30}},
		},
	}
	post := BracketSlice{
		Benchmark: &evaluation.BenchmarkSummary{Tier: "capability-measured", DecodeTokensPerSecond: 410},
		Evals: []evaluation.EvalSummary{
			{Suite: "store/mmlu", Metrics: map[string]float64{"accuracy": 0.46}, Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize}, ExecutionKnown: true, Comparison: comparison, Record: testutil.ArtifactID(t, artifact.KindEvaluation, "post-mmlu")},
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
	ctx := t.Context()
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
