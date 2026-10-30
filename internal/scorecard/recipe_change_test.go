package scorecard

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestAServedModelsDeclineIsFiledAcrossARecipeChange replays Qwen3.8 Flash Next's first output across the records of
// 2026-10-25 and 2026-10-28: two readings on one recipe, then one 54% worse on the recipe a new activation made, beside
// a model whose readings judge its own cell. The worse reading is a decline, judged against the newest reading under the
// earlier recipe and allowed the share of its value that the column's judged cell is allowed, its own readings being too
// few to say how far it moves; a move within that share files nothing. The decline is keyed without its recipe, so the
// cell declined under a later recipe is found under the same key, and its key with the recipe in it, which a row filed
// before the key left the recipe out carries, still finds the cell. A column with no judged cell leaves a cell with too
// few readings unknown, as before.
func TestAServedModelsDeclineIsFiledAcrossARecipeChange(t *testing.T) {
	t.Parallel()
	flash, steady := testutil.ArtifactID(t, artifact.KindModel, "flash"), testutil.ArtifactID(t, artifact.KindModel, "steady")
	recipe := func(name string) Contract {
		return Contract{Metric: "time to prefill", Origin: originServing, Scope: testutil.ArtifactID(t, artifact.KindRecipe, name)}
	}
	first, activated, later := recipe("first"), recipe("activated"), recipe("later")
	steadyRecipe := recipe("steady")
	cell := func(value float64, contract Contract) Cell {
		return Cell{Value: value, Unit: "ms", Contract: contract}
	}
	row := func(model artifact.ID, value float64, contract Contract) Row {
		return measuredFixture(model, map[metric.Name]Cell{metric.FirstOutput: cell(value, contract)})
	}
	// records are six readings of the steady model, the flash model's two under its first recipe in the last two, and
	// then the newest record's readings of each.
	records := func(peer bool, newest float64, contract Contract) []Record {
		var built []Record
		for index, value := range []float64{100, 101, 99, 100, 101, 100} {
			record := Record{Commit: "earlier"}
			if peer {
				record.Rows = append(record.Rows, row(steady, value, steadyRecipe))
			}
			if index >= 4 {
				record.Rows = append(record.Rows, row(flash, []float64{936.2, 1128.1}[index-4], first))
			}
			built = append(built, record)
		}
		newestRecord := Record{Commit: "newest", Rows: []Row{row(flash, newest, contract)}}
		if peer {
			newestRecord.Rows = append(newestRecord.Rows, row(steady, 100, steadyRecipe))
		}
		return append(built, newestRecord)
	}
	declines := func(records []Record) []Gap {
		newest := records[len(records)-1]
		return slices.DeleteFunc(newest.Gaps(Reference(records)), func(gap Gap) bool { return gap.kind != string(regressed) })
	}

	filed := declines(records(true, 1742.1, activated))
	if len(filed) != 1 || !strings.Contains(filed[0].Detail, "measured under "+activated.Scope.String()) ||
		!strings.Contains(filed[0].Detail, "2 earlier readings on its contract") || !strings.Contains(filed[0].Detail, "judged first-output cells") {
		t.Fatalf("1742 ms under a new recipe after 936 and 1128 under the first = %+v, want one decline judged across the recipes by the column's share", filed)
	}
	if within := declines(records(true, 1180, activated)); len(within) != 0 {
		t.Errorf("1180 ms under a new recipe after 1128 = %+v, want none: the column's judged cell allows 8%% of a value", within)
	}
	if again := declines(records(true, 1800, later)); len(again) != 1 || again[0].Key != filed[0].Key {
		t.Errorf("the decline under a later recipe = %+v, want it filed under the key of the decline under the activated recipe, %s", again, filed[0].Key)
	}
	flashRow := records(true, 1742.1, activated)[6].Rows[0]
	withRecipe := recipeRegressionKey(flashRow, metric.FirstOutput, flashRow.Cells[metric.FirstOutput])
	if restored, found := Restored([]Row{flashRow}, withRecipe, 1218); !found || restored {
		t.Errorf("the cell found by its key with the recipe in it = (restored %t, found %t), want found and not restored", restored, found)
	}
	if withRecipe == filed[0].Key || !slices.Equal(filed[0].Aliases, []string{withRecipe}) {
		t.Errorf("the decline's other keys = %q beside %s, want the key with the activated recipe in it", filed[0].Aliases, filed[0].Key)
	}
	if alone := declines(records(false, 1742.1, activated)); len(alone) != 0 {
		t.Errorf("1742 ms with no judged cell in its column = %+v, want it unknown", alone)
	}
}
