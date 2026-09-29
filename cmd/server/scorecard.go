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

// comparedRecords are the published scorecards a history compares: the
// previous and the newest.
const comparedRecords = 2

// scorecardHistory reads what moved between the two newest scorecards
// published beside the store at storeRoot, as the scorecard package compares
// them; there is none until two are published or without a store.
func scorecardHistory(storeRoot string) llamaserver.ScorecardHistoryReader {
	if storeRoot == "" {
		return nil
	}
	return func() (*llamaserver.ScorecardHistory, error) {
		records, err := scorecard.LatestRecords(scorecard.RecordsDirectory(storeRoot), comparedRecords)
		if err != nil || len(records) < comparedRecords {
			return nil, err
		}
		history := &llamaserver.ScorecardHistory{Previous: records[0].Commit, Current: records[1].Commit, Moves: []llamaserver.ScorecardMove{}}
		for _, move := range scorecard.Moves(records[0], records[1]) {
			history.Moves = append(history.Moves, llamaserver.ScorecardMove{Name: move.Name, Task: move.Task, Kind: string(move.Kind), Metric: string(move.Column),
				Before: move.Before, After: move.After, Unit: move.Unit, Reason: move.Reason, Improved: move.Improved()})
		}
		return history, nil
	}
}
