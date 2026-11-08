package loop

import (
	"slices"
	"testing"
)

// TestAChangeRemeasuresOnlyTheCellsItReaches holds the cells a change re-measures to those whose measured path reads
// what it changed: the owner asked that a change measure only what it can move, not every model each time, and the
// lead chose the models by hand. A change to the forecast family's loader reaches the forecast cells alone, one to the
// shared inference package every text cell, and a change to a document no cell reads, none.
func TestAChangeRemeasuresOnlyTheCellsItReaches(t *testing.T) {
	t.Parallel()
	cells := map[string][]string{
		"chronos forecast": {"internal/seriesforecast", "internal/tensor"},
		"timesfm forecast": {"internal/seriesforecast", "internal/tensor"},
		"qwen inference":   {"internal/inference", "internal/tensor"},
		"gemma inference":  {"internal/inference", "internal/tensor"},
		"tabpfn tabular":   {"internal/tabularicl", "internal/tensor"},
	}
	for _, reach := range []struct {
		changed []string
		want    []string
	}{
		{[]string{"internal/seriesforecast/load.go"}, []string{"chronos forecast", "timesfm forecast"}},
		{[]string{"internal/inference/session.go", "README.md"}, []string{"gemma inference", "qwen inference"}},
		{[]string{"internal/tensor/matmul.go"}, []string{"chronos forecast", "gemma inference", "qwen inference", "tabpfn tabular", "timesfm forecast"}},
		{[]string{"docs/plan.json", "skill.md"}, nil},
	} {
		if got := ReachedCells(cells, reach.changed); !slices.Equal(got, reach.want) {
			t.Errorf("a change to %v re-measures %v, want %v", reach.changed, got, reach.want)
		}
	}
}
