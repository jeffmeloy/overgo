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
// earlier record that measured the commit the promoted branch holds. Without
// one there is no baseline: an earlier record of another commit is not what
// users run, so it judges nothing.
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
	return nil, nil
}

// Withheld reports why the record may not promote its commit, or "" when it
// may: promotion needs a baseline measured on the promoted branch, a
// held-out quality on every row it applies to, and every held-out quality
// the baseline measured holding, neither regressed nor Unknown. A task whose
// held-out evaluation is only proposed reads not applicable, so it withholds
// nothing until its evaluation lands.
func (record Record) Withheld(baseline *Record) string {
	if baseline == nil {
		return "no earlier scorecard measured the commit " + PromotedBranch + " holds"
	}
	for _, row := range record.Rows {
		if cell := row.Cells[metric.Quality]; !cell.NotApplicable && !cell.Measured() {
			return fmt.Sprintf("%s has no %s held-out quality: %s", row.Name, row.Task, cell.Reason)
		}
	}
	for _, gap := range record.Gaps(baseline) {
		if gap.kind == string(regressed) || gap.kind == string(Unknown) {
			return gap.Title + ": " + strings.TrimSpace(gap.Detail)
		}
	}
	return ""
}

// Verdict is how a record holds a cell its baseline measured: holding,
// regressed, or Unknown, the verdict a consumer outside the package tells
// apart from a measured change.
type Verdict string

const (
	holding   Verdict = "holding"
	regressed Verdict = "regressed"
	// Unknown is the verdict on a cell nothing compares: the record lost its
	// row or the cell, or measured it on another contract. It withholds
	// promotion until the record measures the cell on the baseline's
	// contract again, or the promoted branch moves to a commit whose
	// scorecard adopts the new scope.
	Unknown Verdict = "unknown"
)

// Pair is one model and task in a baseline and a record: its row in each,
// nil where that record has none.
type Pair struct{ Before, After *Row }

// Pairs pairs the record's rows with the baseline's by model and task, never
// by display name, so a renamed model keeps its baseline and two models
// sharing a name never trade theirs: the record's rows in order, each beside
// its baseline row, then the baseline's rows the record lacks.
func (record Record) Pairs(baseline Record) []Pair {
	pairs := make([]Pair, len(record.Rows))
	for index := range record.Rows {
		pairs[index].After = &record.Rows[index]
	}
	for index, row := range baseline.Rows {
		at := slices.IndexFunc(record.Rows, func(candidate Row) bool { return candidate.Model == row.Model && candidate.Task == row.Task })
		if at < 0 {
			at = len(pairs)
			pairs = append(pairs, Pair{})
		}
		pairs[at].Before = &baseline.Rows[index]
	}
	return pairs
}

// Compare reports how the record's row holds the baseline row's cell in
// column, and why when it does not hold. A cell the baseline did not
// measure holds, since the baseline asks nothing of it: a newly measured
// metric or row never blocks. A measured one has regressed when the record
// measured it on the same contract and direction and it fell in that
// direction; it is Unknown when the record lost the row or the cell,
// measured it on another contract, or the baseline cell has no contract.
func (pair Pair) Compare(column metric.Name) (Verdict, string) {
	if pair.Before == nil || !pair.Before.Cells[column].Measured() {
		return holding, ""
	}
	if pair.After == nil {
		return Unknown, "the record has no row for this model and task"
	}
	then, now := pair.Before.Cells[column], pair.After.Cells[column]
	switch {
	case !now.Measured():
		return Unknown, "no longer measured: " + now.Reason
	case then.Contract == (Contract{}):
		return Unknown, "the baseline cell was published without a contract"
	case now.Contract != then.Contract:
		return Unknown, fmt.Sprintf("measured %+v, the baseline %+v", now.Contract, then.Contract)
	case now.Direction != then.Direction:
		return Unknown, fmt.Sprintf("improves in direction %q, the baseline in %q", now.Direction, then.Direction)
	}
	if advantage, directed := now.Direction.Advantage(now.Value, then.Value); directed && advantage < 0 {
		return regressed, fmt.Sprintf("%.4g to %.4g (%s)", then.Value, now.Value, now.Label)
	}
	return holding, ""
}

// The kinds of gap besides the regressed and Unknown verdicts. Their names
// order the rows: failures a user met, then held-out qualities that fell or
// that nothing compares, which refuse promotion, then measurements to take,
// then held-out evaluations to define.
const (
	gapFailing    = "failing"
	gapUnmeasured = "unmeasured"
	gapUnscored   = "unscored"
)

// Gap is the work one task's shortfall on one metric asks for, with every
// row that falls short and its reason or values. Key names the gap the same
// way in every record, so its row is proposed once and its verifier finds
// it again.
type Gap struct {
	Key, Title, Detail string
	kind               string
}

// Gaps lists the record's gaps against the baseline, which may be nil: each
// cell no record measures, each failure-rate above zero, each held-out
// quality the baseline measured that Compare finds regressed or Unknown, and
// each held-out quality that is not applicable because its task's held-out
// evaluation is only proposed.
func (record Record) Gaps(baseline *Record) []Gap {
	var before Record
	if baseline != nil {
		before = *baseline
	}
	grouped := map[string]*Gap{}
	// One gap per kind and task: its metrics are lines of the one row, so
	// a task's missing measurements are one piece of work.
	add := func(kind string, row *Row, column metric.Name, line string) {
		key := strings.Join([]string{"scorecard", kind, string(row.Task)}, "-")
		if grouped[key] == nil {
			grouped[key] = &Gap{Key: key, Title: gapTitle(kind, row.Task), kind: kind}
		}
		grouped[key].Detail += "- " + row.Name + " " + string(column) + ": " + line + "\n"
	}
	for _, pair := range record.Pairs(before) {
		verdict, why := pair.Compare(metric.Quality)
		if pair.After == nil {
			if verdict != holding {
				add(string(verdict), pair.Before, metric.Quality, why)
			}
			continue
		}
		for _, column := range metric.Columns {
			switch cell := pair.After.Cells[column]; {
			case !cell.Measured() && !cell.NotApplicable:
				add(gapUnmeasured, pair.After, column, cell.Reason)
			case column == metric.Quality && verdict != holding:
				add(string(verdict), pair.After, column, why)
			case column == metric.Quality && cell.NotApplicable:
				add(gapUnscored, pair.After, column, cell.Reason)
			case column == metric.Failures && cell.Value > 0:
				add(gapFailing, pair.After, column, cell.Label)
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

// gapTitle names the work a gap asks for.
func gapTitle(kind string, task recipe.Task) string {
	switch kind {
	case string(regressed):
		return fmt.Sprintf("Recover %s held-out quality, which fell below the baseline scorecard", task)
	case string(Unknown):
		return fmt.Sprintf("Measure %s held-out quality again on the baseline scorecard's models and contracts", task)
	case gapFailing:
		return fmt.Sprintf("Fix the %s attempts that failed", task)
	case gapUnscored:
		return fmt.Sprintf("Define the held-out evaluation that measures %s quality", task)
	default:
		return fmt.Sprintf("Measure %s's unmeasured cells", task)
	}
}
