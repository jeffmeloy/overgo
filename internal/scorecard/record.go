package scorecard

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// PromotedBranch is the branch users run: the loop moves it, only forward,
// to a commit whose scorecard holds against its baseline.
const PromotedBranch = "refs/heads/main"

// A published record is named by the time it was published, so the
// directory lists its records oldest first; the directory sits beside the
// store, outside the module.
const (
	recordDirectory                 = "scorecards"
	recordPrefix                    = "scorecard-"
	recordSuffix                    = ".json"
	recordTime                      = "20060102T150405Z"
	recordDirectoryMode fs.FileMode = 0o755
	recordMode          fs.FileMode = 0o644
)

// Record is one published scorecard: the commit it measured and every row.
// Name is the file it was read from.
type Record struct {
	Commit string `json:"commit"`
	Rows   []Row  `json:"rows"`
	Name   string `json:"-"`
}

// Directory is where the scorecards of the store at root are published.
func Directory(root string) string {
	return filepath.Join(filepath.Dir(root), recordDirectory)
}

// Publish writes the record as the directory's newest, named by when it was
// published, and returns its path. A record with a cell that no record
// measures and no reason explains is refused, and nothing is written.
func Publish(directory string, published time.Time, record Record) (string, error) {
	if err := record.explained(); err != nil {
		return "", err
	}
	if err := os.MkdirAll(directory, recordDirectoryMode); err != nil {
		return "", err
	}
	path := filepath.Join(directory, recordPrefix+published.UTC().Format(recordTime)+recordSuffix)
	return path, jsonfile.Write(path, record, recordMode)
}

// Records reads every published record, oldest first; an absent directory
// has none. A record with an unexplained cell is refused, so nothing is
// judged or proposed from it.
func Records(directory string) ([]Record, error) {
	entries, err := os.ReadDir(directory)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var records []Record
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), recordPrefix) || !strings.HasSuffix(entry.Name(), recordSuffix) {
			continue
		}
		record := Record{Name: entry.Name()}
		if err := jsonfile.DecodeStrict(filepath.Join(directory, entry.Name()), &record); err != nil {
			return nil, err
		}
		if err := record.explained(); err != nil {
			return nil, fmt.Errorf("%s: %w", entry.Name(), err)
		}
		records = append(records, record)
	}
	return records, nil
}

// explained refuses a row without a cell for every column, and a cell that
// neither names the record that measured it nor states why none does.
func (record Record) explained() error {
	for _, row := range record.Rows {
		for _, column := range metric.Columns {
			if cell := row.Cells[column]; !cell.Source.Valid() && cell.Reason == "" {
				return fmt.Errorf("scorecard: %s %s, %s: no record measures the cell and no reason explains it", row.Name, row.Task, column)
			}
		}
	}
	return nil
}

// Baseline is the record the newest one is judged against: the newest
// earlier record that measured the commit the promoted branch holds, else
// the one just before it. One record or none has no baseline.
func Baseline(ctx context.Context, root string, records []Record) (*Record, error) {
	if len(records) <= 1 {
		return nil, nil
	}
	held, err := gitauthority.Query(ctx, root, "for-each-ref", "--format=%(objectname)", PromotedBranch)
	if err != nil {
		return nil, err
	}
	earlier := records[:len(records)-1]
	for index, record := range slices.Backward(earlier) {
		if record.Commit == strings.TrimSpace(string(held)) {
			return &earlier[index], nil
		}
	}
	return &earlier[len(earlier)-1], nil
}

// The kinds of gap. Their names order the rows: failures a user met, then
// held-out regressions, which refuse promotion, then measurements to take.
const (
	gapFailing    = "failing"
	gapRegressed  = "regressed"
	gapUnmeasured = "unmeasured"
)

// Gap is the work one task's shortfall on one metric asks for, with every
// row that falls short and its reason or values. Key names the gap the same
// way in every record, so its row is proposed once and its verifier finds
// it again.
type Gap struct {
	Key, Title, Detail string
	kind               string
}

// Regressed reports a held-out quality below the baseline's: the gap that
// refuses promotion.
func (gap Gap) Regressed() bool { return gap.kind == gapRegressed }

// Gaps lists the record's gaps against the baseline, which may be nil: each
// cell no record measures, each failure-rate above zero, and each held-out
// quality below the baseline row's cell that measured the same metric on the
// same suite.
func (record Record) Gaps(baseline *Record) []Gap {
	before := map[string]Row{}
	if baseline != nil {
		for _, row := range baseline.Rows {
			before[row.Name+" "+string(row.Task)] = row
		}
	}
	grouped := map[string]*Gap{}
	// One gap per kind and task: its metrics are lines of the one row, so
	// a task's missing measurements are one piece of work.
	add := func(kind string, row Row, column metric.Name, line string) {
		key := strings.Join([]string{"scorecard", kind, string(row.Task)}, "-")
		if grouped[key] == nil {
			grouped[key] = &Gap{Key: key, Title: gapTitle(kind, row.Task), kind: kind}
		}
		grouped[key].Detail += "- " + row.Name + " " + string(column) + ": " + line + "\n"
	}
	for _, row := range record.Rows {
		for _, column := range metric.Columns {
			cell, then := row.Cells[column], before[row.Name+" "+string(row.Task)].Cells[column]
			switch {
			case cell.NotApplicable:
			case !cell.Measured():
				add(gapUnmeasured, row, column, cell.Reason)
			case column == metric.Failures && cell.Value > 0:
				add(gapFailing, row, column, cell.Label)
			case column == metric.Quality && fell(cell, then):
				add(gapRegressed, row, column, fmt.Sprintf("%.4g to %.4g (%s)", then.Value, cell.Value, cell.Label))
			}
		}
	}
	gaps := make([]Gap, 0, len(grouped))
	for _, gap := range grouped {
		gaps = append(gaps, *gap)
	}
	slices.SortFunc(gaps, func(left, right Gap) int { return strings.Compare(left.Key, right.Key) })
	return gaps
}

// fell reports a measured held-out quality worse than the baseline cell
// that measured the same metric on the same suite, in the metric's own
// direction.
func fell(now, then Cell) bool {
	advantage, directed := now.Direction.Advantage(now.Value, then.Value)
	return directed && advantage < 0 && then.Source.Valid() && then.Label == now.Label && then.Direction == now.Direction
}

// gapTitle names the work a gap asks for.
func gapTitle(kind string, task recipe.Task) string {
	switch kind {
	case gapRegressed:
		return fmt.Sprintf("Recover %s held-out quality, which fell below the baseline scorecard", task)
	case gapFailing:
		return fmt.Sprintf("Fix the %s attempts that failed", task)
	default:
		return fmt.Sprintf("Measure %s's unmeasured cells", task)
	}
}
