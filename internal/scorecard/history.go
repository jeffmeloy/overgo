package scorecard

import (
	"errors"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/jsonfile"
	"overgo/internal/scorecard/metric"
)

// A published scorecard record is named by the time it was published, so the
// directory lists its records oldest first; the directory sits beside the
// store, outside the module.
const (
	recordsDirectory = "scorecards"
	recordPrefix     = "scorecard-"
	recordSuffix     = ".json"
)

// RecordsDirectory is where the scorecards of the store at storeRoot are
// published.
func RecordsDirectory(storeRoot string) string {
	return filepath.Join(filepath.Dir(storeRoot), recordsDirectory)
}

// Record is one published scorecard: the commit it measured and every row's
// cells.
type Record struct {
	Commit string        `json:"commit"`
	Rows   []RecordedRow `json:"rows"`
}

// RecordedRow is one row as published, what a later run compares with.
type RecordedRow struct {
	Name  string                       `json:"name"`
	Task  string                       `json:"task"`
	Cells map[metric.Name]RecordedCell `json:"cells"`
}

// RecordedCell is a cell's measured value and unit, or the reason it has
// none and whether the metric does not apply to the task.
type RecordedCell struct {
	Value         float64 `json:"value,omitzero"`
	Unit          string  `json:"unit,omitzero"`
	Reason        string  `json:"reason,omitzero"`
	NotApplicable bool    `json:"not_applicable,omitzero"`
}

// RecordOf is the rows as the record a publish writes.
func RecordOf(commit string, rows []Row) Record {
	record := Record{Commit: commit}
	for _, row := range rows {
		recorded := RecordedRow{Name: row.Name, Task: string(row.Task), Cells: map[metric.Name]RecordedCell{}}
		for column, cell := range row.Cells {
			recorded.Cells[column] = RecordedCell{Value: cell.Value, Unit: cell.Unit, Reason: cell.Reason, NotApplicable: cell.NotApplicable}
		}
		record.Rows = append(record.Rows, recorded)
	}
	return record
}

// RecordName is the file a record published under stamp is written to.
func RecordName(stamp string) string { return recordPrefix + stamp + recordSuffix }

// LatestRecords reads up to the newest count records of directory, oldest
// first; an absent directory or one without records has none.
func LatestRecords(directory string, count int) ([]Record, error) {
	entries, err := os.ReadDir(directory)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var names []string
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), recordPrefix) && strings.HasSuffix(entry.Name(), recordSuffix) {
			names = append(names, entry.Name())
		}
	}
	slices.Sort(names)
	records := []Record{}
	for _, name := range names[max(0, len(names)-count):] {
		var record Record
		if err := jsonfile.DecodeStrict(filepath.Join(directory, name), &record); err != nil {
			return nil, err
		}
		records = append(records, record)
	}
	return records, nil
}

// MoveKind is how a row or cell moved between two records.
type MoveKind string

const (
	// MoveChanged marks a cell both records measure in one unit whose value changed.
	MoveChanged MoveKind = "changed"
	// MoveMeasured marks a cell that gained its measure.
	MoveMeasured MoveKind = "measured"
	// MoveUnmeasured marks a cell that lost its measure; Reason says why.
	MoveUnmeasured MoveKind = "unmeasured"
	// MoveAdded marks a row that is new.
	MoveAdded MoveKind = "added"
	// MoveRemoved marks a row that is gone.
	MoveRemoved MoveKind = "removed"
)

// Move is one thing that moved from a previous record to the current one: a
// row that came or went, or one of its cells.
type Move struct {
	Name   string      `json:"name"`
	Task   string      `json:"task"`
	Kind   MoveKind    `json:"kind"`
	Column metric.Name `json:"column,omitzero"`
	Before float64     `json:"before,omitzero"`
	After  float64     `json:"after,omitzero"`
	Unit   string      `json:"unit,omitzero"`
	Reason string      `json:"reason,omitzero"`
}

// higherIsBetter are the columns a user wants to rise; every other column is
// a cost a user wants to fall.
var higherIsBetter = []metric.Name{metric.Throughput, metric.Quality}

// Improved reports whether a changed cell moved the way a user wants.
func (move Move) Improved() bool {
	if move.Kind != MoveChanged {
		return false
	}
	return (move.After > move.Before) == slices.Contains(higherIsBetter, move.Column)
}

// Moves lists what moved from the previous record to the current, rows in
// the current record's order, their cells in column order, then the gone
// rows by name and task.
func Moves(previous, current Record) []Move {
	key := func(row RecordedRow) string { return row.Name + " " + row.Task }
	before := map[string]RecordedRow{}
	for _, row := range previous.Rows {
		before[key(row)] = row
	}
	var moves []Move
	for _, row := range current.Rows {
		old, found := before[key(row)]
		delete(before, key(row))
		if !found {
			moves = append(moves, Move{Name: row.Name, Task: row.Task, Kind: MoveAdded})
			continue
		}
		for _, column := range metric.Columns {
			now, then := row.Cells[column], old.Cells[column]
			move := Move{Name: row.Name, Task: row.Task, Column: column, Before: then.Value, After: now.Value, Unit: now.Unit}
			switch {
			case now.Reason == "" && then.Reason != "":
				move.Kind = MoveMeasured
			case now.Reason != "" && then.Reason == "":
				move.Kind, move.Reason = MoveUnmeasured, now.Reason
			case now.Reason == "" && now.Unit == then.Unit && now.Value != then.Value:
				move.Kind = MoveChanged
			default:
				continue
			}
			moves = append(moves, move)
		}
	}
	for _, gone := range slices.Sorted(maps.Keys(before)) {
		row := before[gone]
		moves = append(moves, Move{Name: row.Name, Task: row.Task, Kind: MoveRemoved})
	}
	return moves
}
