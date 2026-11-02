package scorecard

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// TestASlowQualityDeclineIsFiledAgainstItsLevel: a model that loses one case a run never loses enough from one reading to
// the next to be filed. Judged against the best of its newest readings and by that reading's cases, five cases gone stay
// unfiled, and six, the fewest the paired test lets stand, are filed on each of its qualities, the headline and the others.
func TestASlowQualityDeclineIsFiledAgainstItsLevel(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root, store, model, read := pairedReadings(t)
	var records []Record
	for lost, commit := range []string{"a", "b", "c", "d", "e", "f", "1"} {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = inapplicable("not measured in this fixture")
		}
		cells[metric.Quality] = read(commit, 0, 10+lost)
		second := cells[metric.Quality]
		second.Contract.Suite = "second"
		records = append(records, Record{Commit: commit, Rows: []Row{{
			Model: model, Name: "paired", Task: recipe.TaskInference, Cells: cells, Heldout: []Cell{second},
		}}})
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	// declines is what the record's quality gaps come to against a reference, after Supported has judged them by their cases.
	declines := func(record Record, reference *Record) ([]Gap, []string) {
		t.Helper()
		gaps := slices.DeleteFunc(record.Gaps(reference), func(gap Gap) bool { return gap.quality == nil })
		if len(gaps) != len(record.Rows[0].Measured(metric.Quality)) {
			t.Fatalf("%s moved %d qualities: %+v", record.Commit, len(gaps), gaps)
		}
		return Supported(ctx, root, gaps)
	}
	last := len(records) - 1
	if kept, dropped := declines(records[last], Earlier(records)); len(kept) != 0 || !strings.Contains(dropped[0], "1 of the 40 cases") {
		t.Fatalf("one case against the reading before it: kept %+v and dropped %q, want it dropped", kept, dropped)
	}
	if kept, dropped := declines(records[last-1], Reference(records[:last])); len(kept) != 0 || !strings.Contains(dropped[0], "5 of the 40 cases") {
		t.Fatalf("five cases gone from the level: kept %+v and dropped %q, want them dropped", kept, dropped)
	}
	kept, dropped := declines(records[last], Reference(records))
	if len(kept) != 2 || len(dropped) != 0 {
		t.Fatalf("six cases gone from the level: kept %+v and dropped %q, want both qualities filed", kept, dropped)
	}
	// A quality states no unit of its own, and the trades file refuses a trade with none.
	if trade, err := Declare(records[last].Rows, kept[0].Key, "what it bought"); err != nil || trade.Unit != shareMetric {
		t.Fatalf("the decline declared as a trade reads %+v (%v)", trade, err)
	}
	for _, gap := range kept {
		if !strings.Contains(gap.Detail, "6 of the 40 cases") || gap.Target != records[0].Rows[0].Cells[metric.Quality].Value {
			t.Fatalf("the decline reads %+v, want it judged from the first reading", gap)
		}
	}
}
