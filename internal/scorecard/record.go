package scorecard

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
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

// Comparison is a cell the baseline measured that the record does not hold:
// its verdict, regressed or Unknown, and why.
type Comparison struct {
	Verdict Verdict
	Why     string
}

// Compare reports each cell of column the baseline row measured that the
// record's row does not hold; none when every one holds. A cell the baseline
// did not measure holds, since the baseline asks nothing of it: a newly
// measured metric, suite or row never blocks. Each of the baseline row's
// held-out qualities, the headline and every other, is compared on its own
// contract, read from whichever of the record's row's held-out qualities
// measured it, so neither a new headline nor a new order drops one.
func (pair Pair) Compare(column metric.Name) []Comparison {
	if pair.Before == nil {
		return nil
	}
	required := pair.Before.Measured(column)
	if pair.After == nil && len(required) != 0 {
		return []Comparison{{Unknown, "the record has no row for this model and task"}}
	}
	var comparisons []Comparison
	for _, then := range required {
		now, measured := pair.After.Cells[column], pair.After.Measured(column)
		if at := slices.IndexFunc(measured, func(other Cell) bool { return other.Contract == then.Contract }); at >= 0 {
			now = measured[at]
		}
		if verdict, why := holds(then, now); verdict != holding {
			comparisons = append(comparisons, Comparison{verdict, why})
		}
	}
	return comparisons
}

// holds reports how now holds the baseline's measured cell then, and why
// when it does not. It has regressed when now measured it on the same
// contract and direction and it fell in that direction by more than float32
// arithmetic resolves; it is Unknown when now is not measured, measured only
// another contract, or then has no contract.
func holds(then, now Cell) (Verdict, string) {
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
	if fell(now.Direction, now.Value, then.Value) {
		return regressed, fmt.Sprintf("%.9g to %.9g (%s)", then.Value, now.Value, now.Label)
	}
	return holding, ""
}

// fell reports whether value is worse than reference in direction by more
// than float32 arithmetic resolves; a value with no direction never falls.
func fell(direction runrecord.Direction, value, reference float64) bool {
	advantage, directed := direction.Advantage(value, reference)
	return directed && -advantage > float32Precision*max(math.Abs(value), math.Abs(reference))
}

// float32Precision is the relative precision of the float32 logits every
// held-out quality is computed from. Two values closer than this for their
// size differ in summation order, as a kernel's reduction order does across
// commits, not in what the model answered: a re-measured perplexity moves by
// parts per billion, while one case lost in a million moves an accuracy by
// many times this.
const float32Precision = 0x1p-23

// The kinds of gap besides the regressed and Unknown verdicts. Their names
// order the rows: trained models that fall below their base, then failures
// a user met, then held-out qualities that fell or that nothing compares,
// which refuse promotion, then measurements to take, then held-out
// evaluations to define. A trained model below its base refuses no
// promotion: it judges a training against its base model, not the commit
// against the baseline scorecard.
const (
	gapBelowBase  = "below-base"
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
// quality the baseline measured that Compare finds regressed or Unknown,
// each held-out quality that is not applicable because its task's held-out
// evaluation is only proposed, and each trained model's held-out quality
// that fell below its base model's on the verdict that judged them.
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
		compared := pair.Compare(metric.Quality)
		propose := func(row *Row) {
			for _, comparison := range compared {
				add(string(comparison.Verdict), row, metric.Quality, comparison.Why)
			}
		}
		if pair.After == nil {
			propose(pair.Before)
			continue
		}
		for _, column := range metric.Columns {
			switch cell := pair.After.Cells[column]; {
			case !cell.Measured() && !cell.NotApplicable:
				add(gapUnmeasured, pair.After, column, cell.Reason)
			case column == metric.Quality && len(compared) != 0:
				propose(pair.After)
			case column == metric.Quality && cell.NotApplicable:
				add(gapUnscored, pair.After, column, cell.Reason)
			case column == metric.Failures && cell.Value > 0:
				add(gapFailing, pair.After, column, cell.Label)
			}
		}
		if cell := pair.After.Cells[metric.Quality]; cell.Measured() && cell.Base != nil && fell(cell.Direction, cell.Value, *cell.Base) {
			add(gapBelowBase, pair.After, metric.Quality, cell.QualityText())
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
	case gapBelowBase:
		return fmt.Sprintf("Make %s beat its base model on held-out quality", task)
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
