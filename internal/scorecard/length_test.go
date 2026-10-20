package scorecard

import (
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestALongerPromptIsTimedAsItsLengthAlone holds the first outputs a chat
// model's row reads at its longer prompts to the attempts that read them. A
// recipe's verified request (5 tokens, 10 ms after a load), a fresh prompt of
// about 4K tokens (300 ms) and one of about 16K (1,500 ms) give the row's
// own first output 10 ms and two longer-prompt cells of 300 and 1,500 ms, each
// on its own contract; a user's 9,000-token prompt of 4,000 ms lies between
// the two lengths and is timed by neither, nor by the row's own cell, and a
// conversation's 4,500-token prompt of which a prompt cache answered 4,000
// read 500 tokens and is no 4K reading. A longer prompt's cell is judged on
// its own readings: after five of 300 ms one of 600 ms is filed as a decline
// of that cell with 300 as its target, and the history says it moved, beside
// the row's own first output, which did not.
func TestALongerPromptIsTimedAsItsLengthAlone(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "chat-model")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "chat-environment")
	served := testutil.ArtifactID(t, artifact.KindRecipe, "chat-recipe")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "lengths/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: environment}, {ID: served}}}); err != nil {
		t.Fatal(err)
	}
	short, long := evaluation.PromptLengths[0], evaluation.PromptLengths[1]
	// Committed oldest first: the verified request on a fresh load, then the two longer prompts and a
	// user's prompt between them on the loaded model.
	for index, attempt := range []struct {
		tokens, cached uint64
		prefill        time.Duration
		loaded         bool
	}{
		{5, 0, 10 * time.Millisecond, true}, {uint64(short) + 1, 0, 300 * time.Millisecond, false}, {uint64(long) + 1, 0, 1500 * time.Millisecond, false},
		{9000, 0, 4 * time.Second, false}, {4500, 4000, 50 * time.Millisecond, false},
	} {
		observation := runrecord.ServingObservation{
			Model: model, Recipe: served, Environment: environment, Task: recipe.TaskInference, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: int64(index + 1), MeasuredNS: uint64(attempt.prefill), Usage: runrecord.ServingUsage{InputTokens: attempt.tokens, CachedTokens: attempt.cached, OutputTokens: 1},
			Hardware: []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: uint64(attempt.prefill)}},
		}
		if attempt.loaded {
			observation.Phases = []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: uint64(time.Millisecond)}}
		}
		observation, err := runrecord.NewServingObservation(observation)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("lengths/" + observation.ID.String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	summaries, err := servingByRecipe(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	observed := summaries[served]
	if cell := firstOutputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, observed); cell.Value != 10 {
		t.Fatalf("the row's first output = %v ms, want 10, its verified request's", cell.Value)
	}
	lengths := lengthCells(recipe.TaskInference, observed)
	if len(lengths) != 2 || lengths[0].Prompt != short || lengths[0].Value != 300 || lengths[1].Prompt != long || lengths[1].Value != 1500 || lengths[0].Contract == lengths[1].Contract {
		t.Fatalf("longer-prompt cells = %+v, want 300 ms at %d tokens and 1500 ms at %d, on two contracts", lengths, short, long)
	}
	if cells := lengthCells(recipe.TaskImageGen, observed); cells != nil {
		t.Fatalf("an image model's longer-prompt cells = %+v, want none", cells)
	}

	headline := Cell{Value: 10, Unit: "ms", Contract: Contract{Metric: "time to prefill", Origin: originServing}}
	record := func(value float64) Record {
		row := measuredFixture(model, map[metric.Name]Cell{metric.FirstOutput: headline})
		// Each record's reading is its own, as a measurement run's is: a source read again is one reading.
		length := lengths[0]
		length.Value, length.Source = value, artifact.ID{}
		row.Lengths = []Cell{length}
		return Record{Rows: []Row{row}}
	}
	history := slices.Repeat([]Record{record(300)}, fewestReadings)
	slower := record(600)
	records := append(slices.Clone(history), slower)
	key := regressionKey(slower.Rows[0], metric.FirstOutput, slower.Rows[0].Lengths[0])
	filed := slices.DeleteFunc(slower.Gaps(Reference(records)), func(gap Gap) bool { return gap.kind != string(regressed) })
	if len(filed) != 1 || filed[0].Key != key || filed[0].Target != 300 {
		t.Fatalf("declines of a 4K first output from 300 to 600 ms = %+v, want the one, with 300 as its target", filed)
	}
	// Readings that move by about 12 ms allow eight times that: 350 ms after them is no decline.
	noisy := []Record{record(300), record(310), record(290), record(305), record(295), record(350)}
	if filed := slices.DeleteFunc(noisy[len(noisy)-1].Gaps(Reference(noisy)), func(gap Gap) bool { return gap.kind != string(regressed) }); len(filed) != 0 {
		t.Fatalf("declines of a 4K first output at 350 ms after readings of 290 to 310 = %+v, want none: its readings allow it", filed)
	}
	if restored, found := Restored(record(300).Rows, key, 300); !restored || !found {
		t.Fatalf("a 4K first output back at 300 ms = (restored %t, found %t), want restored", restored, found)
	}
	moves := Moves(history[0], slower)
	if len(moves) != 1 || moves[0].Prompt != short || moves[0].Kind != moveChanged || moves[0].Before != 300 || moves[0].After != 600 {
		t.Fatalf("moves = %+v, want the 4K first output's alone, from 300 to 600 ms", moves)
	}
}
