package scorecard

import (
	"cmp"
	"slices"

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
// improves, empty where nothing says.
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
// the comparison the promotion uses: rows pair by model and task, a cell
// both records measure moves only where they measured it on one contract,
// and one they measured on different contracts is Unknown, never better or
// worse. Rows come in the current record's order, their cells in column
// order, then the gone rows.
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
			now, then := pair.After.Cells[column], pair.Before.Cells[column]
			move := Move{Name: row.Name, Task: string(row.Task), Column: column, Before: then.Value, After: now.Value, Unit: now.Unit,
				Direction: cmp.Or(now.Direction, columnDirections[column])}
			comparisons := pair.Compare(column)
			unknown := slices.IndexFunc(comparisons, func(compared Comparison) bool { return compared.Verdict == Unknown })
			switch {
			case now.Measured() && !then.Measured():
				move.Kind = moveMeasured
			case then.Measured() && !now.Measured():
				move.Kind, move.Reason = moveUnmeasured, now.Reason
			case !now.Measured():
				continue
			case unknown >= 0:
				move.Kind, move.Reason = moveUnknown, comparisons[unknown].Why
			case now.Value != then.Value:
				move.Kind = moveChanged
			default:
				continue
			}
			moves = append(moves, move)
		}
	}
	return moves
}
