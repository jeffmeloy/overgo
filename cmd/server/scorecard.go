package main

import (
	"context"

	"overgo/internal/overgodb"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	llamaserver "overgo/internal/server"
)

// serverScorecard reads the user scorecard from the store, each row's cells
// in the order a user meets them.
func serverScorecard(ctx context.Context, store *overgodb.Store) ([]llamaserver.ScorecardRow, error) {
	rows, err := scorecard.Build(ctx, store)
	if err != nil {
		return nil, err
	}
	shown := make([]llamaserver.ScorecardRow, 0, len(rows))
	for _, row := range rows {
		cells := make([]llamaserver.ScorecardCell, 0, len(metric.Columns))
		for _, name := range metric.Columns {
			cell, found := row.Cells[name]
			if !found {
				cell.Reason = "the scorecard reads no " + string(name)
			}
			cells = append(cells, llamaserver.ScorecardCell{Metric: string(name), Value: cell.Value, Unit: cell.Unit, Label: cell.Label, Source: cell.Source, Reason: cell.Reason})
		}
		shown = append(shown, llamaserver.ScorecardRow{Model: row.Model, Name: row.Name, Task: string(row.Task), Recipe: row.Recipe, Cells: cells})
	}
	return shown, nil
}
