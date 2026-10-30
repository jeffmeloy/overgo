package scorecard

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestADeclaredTradeIsNotFiledAgain replays Qwen3.8-27B's peak device memory: readings near 22.7 GB, then 24.0 GB once
// its own layer drafted by default, a decline d184ff653 declared as the drafted decode's price and the next run filed
// again. Declared at the value its cell holds, the same readings file nothing; a reading worse than the declared level
// by more than the cell's readings allow is a decline again, to be restored to that level, and says it was declared. The
// trades file names each cell once and says all it holds, and the embedded one reads.
func TestADeclaredTradeIsNotFiledAgain(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "qwen3.8-27b")
	contract := Contract{Metric: "peak device memory", Origin: originServing, Scope: testutil.ArtifactID(t, artifact.KindRecipe, "drafting")}
	row := func(value float64) Row {
		served := measuredFixture(model, map[metric.Name]Cell{metric.PeakMemory: {Value: value, Unit: "bytes", Contract: contract}})
		served.Name = "Qwen3.8-27B"
		return served
	}
	var records []Record
	for _, value := range []float64{22.70e9, 22.73e9, 22.71e9, 22.72e9, 22.70e9, 22.73e9} {
		records = append(records, Record{Commit: "plain", Rows: []Row{row(value)}})
	}
	declines := func(newest float64, trades []Trade) []Gap {
		all := append(slices.Clone(records), Record{Commit: "drafting", Rows: []Row{row(newest)}})
		return slices.DeleteFunc(all[len(all)-1].gaps(Reference(all), trades), func(gap Gap) bool { return gap.kind != string(regressed) })
	}
	filed := declines(23.99e9, nil)
	if len(filed) != 1 {
		t.Fatalf("24.0 GB after readings near 22.7 = %+v, want one decline", filed)
	}
	trade, err := Declare([]Row{row(23.99e9)}, filed[0].Key, "the drafted decode, 31 to 52 percent faster (d184ff653)")
	if err != nil || trade.Level != 23.99e9 || trade.Model != model || trade.Column != metric.PeakMemory || trade.Contract.Scope.Valid() {
		t.Fatalf("the declared trade = %+v, %v; want the cell at 23.99e9 on its contract without its recipe", trade, err)
	}
	if again := declines(23.99e9, []Trade{trade}); len(again) != 0 {
		t.Errorf("the declared level read again = %+v, want nothing filed", again)
	}
	if within := declines(24.0e9, []Trade{trade}); len(within) != 0 {
		t.Errorf("a reading within what the cell's readings allow of the declared level = %+v, want nothing filed", within)
	}
	worse := declines(26e9, []Trade{trade})
	if len(worse) != 1 || worse[0].Target <= trade.Level || !strings.Contains(worse[0].Detail, "declared at") {
		t.Errorf("26 GB past the declared 24.0 = %+v, want a decline to the declared level that says it was declared", worse)
	}
	if _, err := Declare([]Row{row(23.99e9)}, "scorecard-regressed-0000000000000000", "nothing"); err == nil {
		t.Error("declaring a key no cell carries succeeded")
	}
	for name, document := range map[string]string{
		"repeated":   `{"trades":[` + tradeJSON(model) + `,` + tradeJSON(model) + `]}`,
		"incomplete": `{"trades":[` + strings.Replace(tradeJSON(model), `"bought":"it"`, `"bought":" "`, 1) + `]}`,
	} {
		if _, err := parseTrades([]byte(document)); err == nil {
			t.Errorf("a %s trade was read", name)
		}
	}
	if _, err := parseTrades([]byte(`{"trades":[` + tradeJSON(model) + `]}`)); err != nil {
		t.Errorf("a whole trade was refused: %v", err)
	}
	if _, err := DeclaredTrades(); err != nil {
		t.Errorf("the embedded trades: %v", err)
	}
}

// tradeJSON is one whole trade of model, as the trades file holds it.
func tradeJSON(model artifact.ID) string {
	return `{"model":"` + model.String() + `","name":"m","task":"inference","column":"peak-device-memory","contract":{"metric":"peak device memory","origin":"serving"},"level":1,"unit":"bytes","bought":"it"}`
}
