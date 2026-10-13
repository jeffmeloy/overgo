package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestTwoSuitesAreNeverOneMove holds the moves between two records to cells of one contract.
//
// A row's headline held-out quality was 0.75 on one suite. In the next record its headline is
// 0.60 on another suite, and the first suite is still measured beside it at 0.75. Nothing declined:
// the comparison finds the first suite again and it holds. The move between the two headlines is
// then no change of 0.75 to 0.60, which are answers to different questions. It reads unknown, with
// both suites named, and is no gain and no loss.
//
// The same two headlines on one suite are a change, and a first suite that is no longer measured
// at all is unknown by the comparison's own reason, as before.
func TestTwoSuitesAreNeverOneMove(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "two-suites")
	authority := testutil.ArtifactID(t, artifact.KindProfile, "two-suites-comparison")
	quality := func(value float64, suite string) Cell {
		return Cell{Value: value, Unit: "accuracy", Direction: runrecord.DirectionMaximize, Contract: Contract{Metric: "accuracy", Origin: originEvaluation, Suite: suite, Comparison: authority}}
	}
	row := func(headline Cell, beside ...Cell) Row {
		cells := map[metric.Name]Cell{metric.Quality: headline}
		for _, column := range metric.Columns {
			if _, found := cells[column]; !found {
				cells[column] = Cell{Reason: "no record measures it"}
			}
		}
		return Row{Model: model, Name: "two-suites", Task: recipe.TaskInference, Cells: cells, Heldout: beside}
	}
	before := Record{Commit: "a", Rows: []Row{row(quality(0.75, "store/bbh"))}}

	moved := Moves(before, Record{Commit: "b", Rows: []Row{row(quality(0.60, "store/mmlu-pro"), quality(0.75, "store/bbh"))}})
	if len(moved) != 1 || moved[0].Column != metric.Quality || moved[0].Kind != moveUnknown || moved[0].Advantage() != 0 {
		t.Fatalf("a headline on another suite, with the first suite held beside it, moved as %+v; want the one move unknown, no gain and no loss", moved)
	}
	if reason := moved[0].Reason; !strings.Contains(reason, `"store/mmlu-pro"`) || !strings.Contains(reason, `"store/bbh"`) {
		t.Fatalf("the move's reason is %q, want both suites named", reason)
	}

	if same := Moves(before, Record{Commit: "b", Rows: []Row{row(quality(0.60, "store/bbh"))}}); len(same) != 1 || same[0].Kind != moveChanged || same[0].Advantage() >= 0 {
		t.Fatalf("the same headline suite at a lower value moved as %+v, want one change for the worse", same)
	}
	if gone := Moves(before, Record{Commit: "b", Rows: []Row{row(quality(0.60, "store/mmlu-pro"))}}); len(gone) != 1 || gone[0].Kind != moveUnknown {
		t.Fatalf("a first suite that is no longer measured moved as %+v, want it unknown", gone)
	}
}
