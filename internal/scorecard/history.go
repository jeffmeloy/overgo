package scorecard

import (
	"cmp"
	"fmt"
	"slices"

	"overgo/internal/evaluation"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
)

// MoveKind is how a row or cell moved between two records: "changed", a cell
// both records measure on one contract whose value changed; "measured", a cell
// that gained its measure; "unmeasured", one that lost it; "unknown", one both
// measure that nothing compares, as when they measured it on different
// contracts; "added", a row that is new; or "removed", one that is gone.
type MoveKind string

const (
	moveChanged    MoveKind = "changed"
	moveMeasured   MoveKind = "measured"
	moveUnmeasured MoveKind = "unmeasured"
	moveUnknown    MoveKind = "unknown"
	moveAdded      MoveKind = "added"
	moveRemoved    MoveKind = "removed"
)

// Move is one thing that moved from a previous record to the current one: a
// row that came or went, or one of its cells. Reason says why a cell lost its
// measure or has no comparison; Direction is the way the cell's metric
// improves, empty where nothing says; Prompt is the prompt length of a
// longer prompt's first output, zero for every other cell.
type Move struct {
	Name      string
	Task      string
	Kind      MoveKind
	Column    metric.Name
	Before    float64
	After     float64
	Unit      string
	Reason    string
	Direction runrecord.Direction
	Prompt    int
}

// Advantage is how far a changed cell moved toward what a user wants:
// positive when it improved, negative when it got worse, zero when its
// metric names no direction or the move is not a change on one contract.
func (move Move) Advantage() float64 {
	if move.Kind != moveChanged {
		return 0
	}
	advantage, _ := move.Direction.Advantage(move.After, move.Before)
	return advantage
}

// columnDirections are the way each column besides held-out quality
// improves: throughput rises and every cost falls. A held-out quality's cell
// names its own direction.
var columnDirections = map[metric.Name]runrecord.Direction{
	metric.Throughput:  runrecord.DirectionMaximize,
	metric.FirstOutput: runrecord.DirectionMinimize,
	metric.Failures:    runrecord.DirectionMinimize,
	metric.Load:        runrecord.DirectionMinimize,
	metric.PeakMemory:  runrecord.DirectionMinimize,
	metric.SetupSteps:  runrecord.DirectionMinimize,
}

// Moves lists what moved from the previous record to the current, through
// the comparison the cycle files declines by: rows pair by model and task, a cell
// both records measure moves only where they measured it on one contract,
// and one they measured on different contracts is Unknown, never better or
// worse. Rows come in the current record's order, their cells in column
// order and then each longer prompt's first output, then the gone rows.
func Moves(previous, current Record) []Move {
	var moves []Move
	for _, pair := range current.Pairs(previous) {
		row := cmp.Or(pair.After, pair.Before)
		if pair.Before == nil || pair.After == nil {
			kind := moveAdded
			if pair.After == nil {
				kind = moveRemoved
			}
			moves = append(moves, Move{Name: row.Name, Task: string(row.Task), Kind: kind})
			continue
		}
		for _, column := range metric.Columns {
			if move, moved := cellMove(*row, column, pair.After.Cells[column], pair.Before.Cells[column], pair.Compare(column)); moved {
				moves = append(moves, move)
			}
		}
		// A longer prompt's first output moves on its own, beside the row's own.
		for _, length := range evaluation.PromptLengths {
			now, then := pair.After.length(length), pair.Before.length(length)
			if move, moved := cellMove(*row, metric.FirstOutput, now, then, pair.Compare(metric.FirstOutput)); moved {
				move.Prompt = length
				moves = append(moves, move)
			}
		}
	}
	return moves
}

// cellMove is how a row's cell of a column moved from then to now, and false
// where it did not: the comparisons are the column's, of which those of the
// cell's own prompt length say it has no comparison.
func cellMove(row Row, column metric.Name, now, then Cell, comparisons []Comparison) (Move, bool) {
	move := Move{Name: row.Name, Task: string(row.Task), Column: column, Before: then.Value, After: now.Value, Unit: now.Unit,
		Direction: cmp.Or(now.Direction, columnDirections[column])}
	unknown := slices.IndexFunc(comparisons, func(compared Comparison) bool {
		return compared.Verdict == Unknown && compared.Then.Prompt == then.Prompt
	})
	switch {
	case now.Measured() && !then.Measured():
		move.Kind = moveMeasured
	case then.Measured() && !now.Measured():
		move.Kind, move.Reason = moveUnmeasured, now.Reason
	case !now.Measured():
		return move, false
	case unknown >= 0:
		move.Kind, move.Reason = moveUnknown, comparisons[unknown].Why
	case now.Contract != then.Contract:
		// The two headline cells are of two contracts, as when a row's headline suite changed and
		// the earlier one is still measured beside it. Their difference is no move of either.
		move.Kind, move.Reason = moveUnknown, ContractDifference(now.Contract, then.Contract)
	case now.Value != then.Value:
		move.Kind = moveChanged
	default:
		return move, false
	}
	return move, true
}

// length is the row's first output at a fresh prompt of length tokens, a
// gap where none was read.
func (row Row) length(length int) Cell {
	if at := slices.IndexFunc(row.Lengths, func(cell Cell) bool { return cell.Prompt == length }); at >= 0 {
		return row.Lengths[at]
	}
	return gap(fmt.Sprintf("no attempt read a fresh prompt of %d tokens", length))
}
