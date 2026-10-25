package main

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestALongerPromptsFirstOutputIsPrintedBesideTheRows holds the command's
// table and list of what moved to a chat model's first output at its longer
// prompts. The row's first output reads 40 ms at its verified request, then
// 3.1 s at a 4,096-token prompt. Between two records the 4K cell moves from
// 3.1 to 2.5 s and a 16K cell is newly measured, each printed under its
// prompt, and the row's own first output, which held, prints nothing.
func TestALongerPromptsFirstOutputIsPrintedBesideTheRows(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "long-prompts")
	timing := func(value float64, prompt int) scorecard.Cell {
		metric := "time to prefill"
		if prompt != 0 {
			metric += " at a fresh prompt"
		}
		return scorecard.Cell{Value: value, Unit: "ms", Prompt: prompt, Contract: scorecard.Contract{Metric: metric, Origin: "serving"}}
	}
	row := func(lengths ...scorecard.Cell) scorecard.Row {
		cells := map[metric.Name]scorecard.Cell{metric.FirstOutput: timing(40, 0)}
		for _, column := range metric.Columns {
			if _, found := cells[column]; !found {
				cells[column] = scorecard.Cell{Reason: "no record measures it"}
			}
		}
		return scorecard.Row{Model: model, Name: "long-prompts", Task: recipe.TaskInference, Cells: cells, Lengths: lengths}
	}
	before := scorecard.Record{Rows: []scorecard.Row{row(timing(3100, 4096))}}
	after := scorecard.Record{Rows: []scorecard.Row{row(timing(2500, 4096), timing(9000, 16384))}}

	var table strings.Builder
	if err := printScorecard(t.Context(), before.Rows, t.TempDir(), &table); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(table.String(), "| 40 ms; 3.1 s at 4096 tokens |") {
		t.Fatalf("table = %q, want the first output's cell to read 40 ms, then 3.1 s at 4096 tokens", table.String())
	}
	moved := scorecardMoves(before, after)
	want := []string{
		"- long-prompts inference, First output at a 4096-token prompt: 3100 to 2500 ms (-600)",
		"- long-prompts inference, First output at a 16384-token prompt: now measured, 9000 ms",
	}
	if !slices.Equal(moved, want) {
		t.Fatalf("moves = %q, want %q", moved, want)
	}
}
