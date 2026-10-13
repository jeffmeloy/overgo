package main

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestAChangedHeadlineSuiteIsPrintedAsNoDifference holds the command's list of what moved between
// two records to cells of one contract, as the scorecard's own list is held
// (internal/scorecard, TestTwoSuitesAreNeverOneMove).
//
// A row's headline held-out quality was 0.75 on one suite and is 0.60 on another, with the first
// suite still measured beside it at 0.75. The line for the cell names both suites and gives no
// pair of values and no difference. The same headlines on one suite are printed as a change.
func TestAChangedHeadlineSuiteIsPrintedAsNoDifference(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "two-suites")
	authority := testutil.ArtifactID(t, artifact.KindProfile, "two-suites-comparison")
	quality := func(value float64, suite string) scorecard.Cell {
		return scorecard.Cell{Value: value, Unit: "accuracy", Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: suite, Comparison: authority}}
	}
	row := func(headline scorecard.Cell, beside ...scorecard.Cell) scorecard.Row {
		cells := map[metric.Name]scorecard.Cell{metric.Quality: headline}
		for _, column := range metric.Columns {
			if _, found := cells[column]; !found {
				cells[column] = scorecard.Cell{Reason: "no record measures it"}
			}
		}
		return scorecard.Row{Model: model, Name: "two-suites", Task: recipe.TaskInference, Cells: cells, Heldout: beside}
	}
	record := func(rows ...scorecard.Row) scorecard.Record { return scorecard.Record{Rows: rows} }
	before := record(row(quality(0.75, "store/bbh")))

	moved := scorecardMoves(before, record(row(quality(0.60, "store/mmlu-pro"), quality(0.75, "store/bbh"))))
	if len(moved) != 1 || !strings.HasPrefix(moved[0], "- two-suites inference, Held-out quality: measured on another contract: ") ||
		!strings.Contains(moved[0], `"store/mmlu-pro"`) || !strings.Contains(moved[0], `"store/bbh"`) || strings.Contains(moved[0], "0.75 to 0.6") {
		t.Fatalf("a headline on another suite, with the first suite held beside it, is printed as %q; want one line that names both suites and no difference", moved)
	}
	if same := scorecardMoves(before, record(row(quality(0.60, "store/bbh")))); len(same) != 1 || !strings.Contains(same[0], "Held-out quality: 0.75 to 0.6 accuracy") {
		t.Fatalf("the same headline suite at a lower value is printed as %q, want its two values", same)
	}
}
