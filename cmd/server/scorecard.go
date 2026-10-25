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
			cells = append(cells, shownCell(name, cell))
		}
		var lengths []llamaserver.ScorecardCell
		for _, length := range row.Lengths {
			lengths = append(lengths, shownCell(metric.FirstOutput, length))
		}
		shown = append(shown, llamaserver.ScorecardRow{Model: row.Model, Name: row.Name, Task: string(row.Task), Recipe: row.Recipe, Cells: cells, Lengths: lengths})
	}
	return shown, nil
}

// shownCell is a cell as the page reads it: its value, unit and source; for a
// held-out quality the metric it measures and the direction it improves in;
// and a trained model's base value, which it reads against.
func shownCell(name metric.Name, cell scorecard.Cell) llamaserver.ScorecardCell {
	return llamaserver.ScorecardCell{Metric: string(name), Value: cell.Value, Unit: cell.Unit, Label: cell.Label, Source: cell.Source, Reason: cell.Reason,
		Measure: cell.Contract.Metric, Direction: string(cell.Direction), Base: cell.Base, Prompt: cell.Prompt}
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
		records, err := scorecard.Records(scorecard.Directory(storeRoot))
		if err != nil || len(records) < comparedRecords {
			return nil, err
		}
		previous, current := records[len(records)-comparedRecords], records[len(records)-1]
		history := &llamaserver.ScorecardHistory{Previous: previous.Commit, Current: current.Commit, Moves: []llamaserver.ScorecardMove{}}
		for _, move := range scorecard.Moves(previous, current) {
			history.Moves = append(history.Moves, llamaserver.ScorecardMove{Name: move.Name, Task: move.Task, Kind: moveKind(move), Metric: string(move.Column),
				Before: move.Before, After: move.After, Unit: move.Unit, Reason: move.Reason, Prompt: move.Prompt})
		}
		return history, nil
	}
}

// moveKind names a move as the page groups it: a change on one contract is
// better or worse by its metric's direction, and stays changed where the
// metric names none.
func moveKind(move scorecard.Move) string {
	switch advantage := move.Advantage(); {
	case advantage > 0:
		return "better"
	case advantage < 0:
		return "worse"
	}
	return string(move.Kind)
}
