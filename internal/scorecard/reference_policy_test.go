package scorecard

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestAReferenceReadsItsOwnPolicysReadings holds the earlier readings a serving reference's allowance is set from to
// its own population: on 2026-11-09 the first mutation run (loop-proof/mutants) inverted the comparisons that choose
// them and no test failed. Of five earlier readings of one row, the reference read under policy A in one environment
// in one direction keeps the one under A and the one from before cells named a policy, and leaves the one under
// policy B, the one in another environment and the one in the other direction.
func TestAReferenceReadsItsOwnPolicysReadings(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "policy reference model")
	environment, elsewhere := testutil.ArtifactID(t, artifact.KindEvidence, "here"), testutil.ArtifactID(t, artifact.KindEvidence, "elsewhere")
	first, second := testutil.ArtifactID(t, artifact.KindProfile, "policy A"), testutil.ArtifactID(t, artifact.KindProfile, "policy B")
	contract := Contract{Metric: "output tokens over decode time", Origin: originServing, Scope: testutil.ArtifactID(t, artifact.KindRecipe, "policy reference recipe")}
	reading := func(name string, value float64, policy, at artifact.ID, direction runrecord.Direction) Cell {
		return Cell{Value: value, Source: testutil.ArtifactID(t, artifact.KindEvidence, name), Environment: at, Policy: policy, Direction: direction, Contract: contract}
	}
	cells := []Cell{
		reading("under A", 1, first, environment, runrecord.DirectionMaximize),
		reading("before policies", 2, artifact.ID{}, environment, runrecord.DirectionMaximize),
		reading("under B", 3, second, environment, runrecord.DirectionMaximize),
		reading("elsewhere", 4, first, elsewhere, runrecord.DirectionMaximize),
		reading("the other way", 5, first, environment, runrecord.DirectionMinimize),
	}
	row := Row{Model: model, Name: "policy reference", Task: recipe.TaskInference}
	var earlier []Record
	for _, cell := range cells {
		held := row
		held.Cells = map[metric.Name]Cell{metric.Throughput: cell}
		earlier = append(earlier, Record{Rows: []Row{held}})
	}
	reference := reading("the reference", 1, first, environment, runrecord.DirectionMaximize)
	read := readingsOf(earlier, row, reference, func(row Row) Cell { return row.Cells[metric.Throughput] }, func(Cell) bool { return true })
	var values []float64
	for _, cell := range read {
		values = append(values, cell.Value)
	}
	if !slices.Equal(values, []float64{1, 2}) {
		t.Fatalf("the reference reads the earlier readings %v, want those under its policy and before policies, 1 and 2", values)
	}
}
