package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAMediaCellRecordsThePresetItRan holds a generated clip's time to what it ran: on 2026-11-08 Wan's cell timed 21
// frames at 32 x 16 in 2 steps and recorded neither, beside a person's request of 81 frames at 832 x 480 in 50. The cell
// carries the preset its measured attempt ran, an older attempt of another preset joins none of its readings, a time
// whose attempt recorded no preset reads as unmeasured, and the README's generation table prints each time with its
// preset, the user-size one too, and nothing for a record whose times name none.
func TestAMediaCellRecordsThePresetItRan(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	model := testutil.ArtifactID(t, artifact.KindModel, "preset model")
	kept, bare := testutil.ArtifactID(t, artifact.KindRecipe, "preset recipe"), testutil.ArtifactID(t, artifact.KindRecipe, "bare recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "preset environment")
	testcheck.Must(store.Commit(ctx, artifact.Batch{Key: "preset/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: kept}, {ID: bare}, {ID: environment}}}))(t)
	small := media.Preset{Width: 32, Height: 16, Frames: 21, Steps: 2, Guidance: 5, Sampler: "flow UniPC", Shift: 5}
	sized := media.Preset{Width: 832, Height: 480, Frames: 81, Steps: 50, Guidance: 5, Sampler: "flow UniPC", Shift: 5}
	attempt := func(label string, recipeID artifact.ID, started int64, took uint64, preset media.Preset, userSize bool) {
		observation := testcheck.Must(runrecord.NewServingObservation(runrecord.ServingObservation{
			Model: model, Recipe: recipeID, Environment: environment, Task: recipe.TaskVideoGen, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: started, MeasuredNS: took, UserSize: userSize, Preset: preset,
			Phases: []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: 1}},
		}))(t)
		testcheck.Must(store.Commit(ctx, testcheck.Must(observation.Batch("preset/"+label))(t)))(t)
	}
	attempt("other preset", kept, 1, 9_000_000_000, media.Preset{Width: 32, Height: 16, Frames: 21, Steps: 4, Sampler: "flow UniPC"}, false)
	attempt("small", kept, 3, 13_000_000_000, small, false)
	attempt("sized", kept, 2, 374_000_000_000, sized, true)
	attempt("no preset", bare, 4, 13_000_000_000, media.Preset{}, false)
	summaries := testcheck.Must(servingByRecipe(ctx, store))(t)
	first := firstOutputCell(recipe.TaskVideoGen, evaluation.BenchmarkSummary{}, false, summaries[kept])
	if first.Preset != small || first.Value != 13_000 {
		t.Fatalf("the clip's time reads %v ms at %+v, want 13000 ms at %+v alone", first.Value, first.Preset, small)
	}
	if unmarked := firstOutputCell(recipe.TaskVideoGen, evaluation.BenchmarkSummary{}, false, summaries[bare]); unmarked.Measured() {
		t.Fatalf("a clip's time whose attempt recorded no preset reads %+v, want it unmeasured", unmarked)
	}
	row := Row{Name: "Wan", Task: recipe.TaskVideoGen, Cells: map[metric.Name]Cell{metric.FirstOutput: first}, Warm: userSizeCells(summaries[kept])}
	want := "| Wan | 13.0 s (32 x 16, 21 frames, 2 steps, flow UniPC, guidance 5, shift 5) | 374.0 s (832 x 480, 81 frames, 50 steps, flow UniPC, guidance 5, shift 5) |\n"
	if table := (Record{Rows: []Row{row}}).GenerationTable(); !strings.HasSuffix(table, want) {
		t.Fatalf("the README's generation table reads\n%s\nwant it to end with\n%s", table, want)
	}
	row.Cells[metric.FirstOutput] = Cell{Value: 13_000}
	if table := (Record{Rows: []Row{row}}).GenerationTable(); table != "" {
		t.Fatalf("a record whose times name no preset gives the table\n%s", table)
	}
}
