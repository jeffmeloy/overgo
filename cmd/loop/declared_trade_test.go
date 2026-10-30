package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
)

// TestADeclaredTradeIsNotFiledAgain holds the loop to the trades the scorecard reads: Qwen3.8-27B's drafting memory,
// declared in d184ff653 and filed again by the run of 2026-10-30, read again at the level it was declared at after
// readings 5% lower files no row, and read 10% past that level files one, to be restored to the declared level.
func TestADeclaredTradeIsNotFiledAgain(t *testing.T) {
	t.Parallel()
	trades, err := scorecard.DeclaredTrades()
	if err != nil {
		t.Fatal(err)
	}
	at := -1
	for index, trade := range trades {
		if trade.Name == "Qwen3.8-27B" && trade.Column == metric.PeakMemory {
			at = index
		}
	}
	if at < 0 {
		t.Fatalf("the trades %+v do not declare Qwen3.8-27B's drafting memory", trades)
	}
	trade := trades[at]
	row := func(value float64) scorecard.Row {
		return scorecard.Row{Model: trade.Model, Name: trade.Name, Task: trade.Task, Cells: map[metric.Name]scorecard.Cell{
			trade.Column: {Value: value, Unit: trade.Unit, Contract: trade.Contract},
		}}
	}
	filed := func(newest float64) int {
		var records []scorecard.Record
		for _, share := range []float64{0.946, 0.947, 0.946, 0.948, 0.947, 0.946} {
			records = append(records, scorecard.Record{Commit: "plain", Rows: []scorecard.Row{row(trade.Level * share)}})
		}
		current := scorecard.Record{Commit: "drafting", Rows: []scorecard.Row{row(newest)}}
		improver := improvement{root: t.TempDir(), plan: emptyPlan(t)}
		if _, err := improver.Propose(current, current.Gaps(scorecard.Reference(append(records, current)))); err != nil {
			t.Fatal(err)
		}
		document, err := plan.Load(improver.plan)
		if err != nil {
			t.Fatal(err)
		}
		// The row's other gaps, such as its path compared with no reference, are not the trade's.
		declines := 0
		for _, item := range document.Items {
			if strings.HasPrefix(item.ID, "scorecard-regressed-") {
				declines++
			}
		}
		return declines
	}
	if added := filed(trade.Level); added != 0 {
		t.Errorf("the declared level read again filed %d row(s), want none", added)
	}
	if added := filed(trade.Level * 1.1); added != 1 {
		t.Errorf("a reading 10%% past the declared level filed %d row(s), want one", added)
	}
}
