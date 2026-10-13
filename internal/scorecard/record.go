package scorecard

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
)

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

// Record is one published scorecard: the publishing commit and every row.
// A cell's producing CodeCommit is separate; publication does not prove currency.
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

// Reference is what the newest record's cells are judged against: its earlier
// measurements (Earlier), with each cell of a column that a repeat of the same work
// reads differently replaced by what the cell's own readings say of it.
//
// A cell's readings are the values the earlier records hold for it on its contract,
// environment and direction, each source once and oldest first, the newest
// referenceWindow of them. Its value is the best level they held, a level being the
// median of levelReadings consecutive readings: one reading far from its neighbours is
// no level, an improvement is one once two readings hold it, and a run of small
// declines is judged against where the cell stood before them, not against the
// reading before the last. What a later reading is allowed is noiseMultiple times the
// median distance between consecutive readings, which a step among them does not
// widen. Fewer than fewestReadings say neither, and such a cell is Unknown, never
// holding. A level leaves with the window, so a decline that stands is a decline until
// then. Nil when no record is earlier.
func Reference(records []Record) *Record {
	reference := Earlier(records)
	if reference == nil {
		return nil
	}
	earlier := records[:len(records)-1]
	for _, row := range reference.Rows {
		for _, column := range repeated {
			cell := row.Cells[column]
			if !cell.Measured() {
				continue
			}
			var readings []float64
			var read Cell
			for _, record := range earlier {
				at := slices.IndexFunc(record.Rows, func(candidate Row) bool { return candidate.Model == row.Model && candidate.Task == row.Task })
				if at < 0 {
					continue
				}
				// The value a later record carries from the same source is the same reading.
				if one := record.Rows[at].Cells[column]; one.Measured() && one.Contract == cell.Contract && one.Environment == cell.Environment && one.Direction == cell.Direction &&
					(len(readings) == 0 || !one.Source.Valid() || one.Source != read.Source || one.Value != read.Value) {
					readings, read = append(readings, one.Value), one
				}
			}
			readings = readings[max(0, len(readings)-referenceWindow):]
			if len(readings) < fewestReadings {
				cell.unjudged = fmt.Sprintf("%d earlier readings on its contract, fewer than the %d that say what it holds and how far it moves", len(readings), fewestReadings)
				row.Cells[column] = cell
				continue
			}
			direction := cmp.Or(cell.Direction, columnDirections[column])
			moves := make([]float64, len(readings)-1)
			for index := range moves {
				moves[index] = math.Abs(readings[index+1] - readings[index])
			}
			cell.Value, cell.allowed = median(slices.Clone(readings[:levelReadings])), noiseMultiple*median(moves)
			for from := 1; from+levelReadings <= len(readings); from++ {
				held := median(slices.Clone(readings[from : from+levelReadings]))
				if advantage, _ := direction.Advantage(held, cell.Value); advantage > 0 {
					cell.Value = held
				}
			}
			row.Cells[column] = cell
		}
	}
	return reference
}

// repeated are the columns a repeat of the same work reads differently: the three
// timings and the device's peak memory. A held-out quality and a count of setup
// steps are what the work gave and repeat to the digit, and the failed share is
// judged by its attempts.
var repeated = []metric.Name{metric.FirstOutput, metric.Throughput, metric.Load, metric.PeakMemory}

// What a cell of a repeated column is judged by (Reference). The four were chosen on
// the 32 records published from 2026-09-30 to 2026-10-13, judged in turn by this code
// with each cell's readings taken as the medians of three that a record publishes
// since 2026-10-13 (a replay the commit that set them describes). At eight times its
// noise the rule files 42 load-time readings on 14 cells, none of them back at its target
// by the cell's next reading: most are the growth of the store's open, which the
// frozen allowances filed 2 of. It files 7 first-output and 2 throughput readings, 1
// of the 9 back at its target by the next reading. At six times it files 17 of those
// two columns with 5 back, and at four times 29 with 7 back. At ten times it files
// 5 with 1 back and loses two of the load-time cells. Those readings were taken
// beside other work, so they are the noisiest the cells will be judged on.
const (
	// referenceWindow is how many of a cell's newest distinct readings judge it: wide
	// enough that eight small declines in a row are judged against the first, and
	// narrow enough that readings taken beside other work, as all before 2026-10-13
	// were, leave it within nine more.
	referenceWindow = 9
	// levelReadings is how many consecutive readings hold a level, the fewest whose
	// median one reading far from the others does not move.
	levelReadings = 3
	// fewestReadings is the fewest readings that judge a cell: four distances between
	// consecutive readings, whose median neither the largest nor the smallest sets.
	fewestReadings = 5
	// noiseMultiple is how many times its noise a reading may stand worse than its level.
	noiseMultiple = 8
)

// Earlier is each cell's newest earlier measurement, which what moved since the
// previous record is read against: the cells of the newest earlier record, each one
// that record did not measure taken from the newest older record that did, so a cell
// is compared with its newest earlier complete measurement and never with nothing. A
// value is never carried forward in place of a missing current one: it only ever
// answers a cell the newest record has. Nil when no record is earlier.
func Earlier(records []Record) *Record {
	if len(records) <= 1 {
		return nil
	}
	earlier := records[:len(records)-1]
	newest := earlier[len(earlier)-1]
	reference := Record{Commit: newest.Commit, Name: newest.Name, Rows: slices.Clone(newest.Rows)}
	for index := range reference.Rows {
		reference.Rows[index].Cells = maps.Clone(reference.Rows[index].Cells)
	}
	for _, older := range slices.Backward(earlier[:len(earlier)-1]) {
		for _, row := range older.Rows {
			at := slices.IndexFunc(reference.Rows, func(candidate Row) bool { return candidate.Model == row.Model && candidate.Task == row.Task })
			if at < 0 {
				continue
			}
			for column, cell := range row.Cells {
				if kept := reference.Rows[at].Cells[column]; !kept.Measured() && !kept.NotApplicable && cell.Measured() {
					reference.Rows[at].Cells[column] = cell
				}
			}
		}
	}
	return &reference
}

// Verdict is how a record's cell compares with its reference: holding, regressed,
// or Unknown, the verdict a consumer outside the package tells apart from a
// measured change. Only a regression asks for work; an Unknown is reported.
type Verdict string

const (
	holding   Verdict = "holding"
	regressed Verdict = "regressed"
	// Unknown is the verdict on a cell nothing compares: the record lost the
	// cell, or measured it on another contract than its reference did. It is
	// reported with its reason and files no row.
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

// Comparison is a cell of the record that does not hold against its reference:
// its verdict, regressed or Unknown, why, and the two cells compared (the
// reference's is empty where the record has no row to compare).
type Comparison struct {
	Verdict   Verdict
	Why       string
	Then, Now Cell
}

// Compare reports each of the reference row's measured cells of column that the
// record's row does not hold; none when every one holds. A cell the reference
// did not measure holds, since the reference asks nothing of it: a newly
// measured metric, suite or row is never a decline. Each of the reference row's
// held-out qualities, the headline and every other, is compared on its own
// contract, read from whichever of the record's row's held-out qualities
// measured it, so neither a new headline nor a new order drops one.
func (pair Pair) Compare(column metric.Name) []Comparison {
	if pair.Before == nil {
		return nil
	}
	required := pair.Before.Measured(column)
	if pair.After == nil && len(required) != 0 {
		return []Comparison{{Verdict: Unknown, Why: "the record has no row for this model and task"}}
	}
	var comparisons []Comparison
	for _, then := range required {
		now, measured := pair.After.Cells[column], pair.After.Measured(column)
		if at := slices.IndexFunc(measured, func(other Cell) bool { return other.Contract == then.Contract }); at >= 0 {
			now = measured[at]
		}
		if verdict, why := holds(column, then, now); verdict != holding {
			comparisons = append(comparisons, Comparison{verdict, why, then, now})
		}
	}
	return comparisons
}

// ContractDifference says in words how two contracts differ, naming each part
// that does: the measure, where its value comes from, the suite and protocol it
// was scored on, the comparison authority, and the recipe it was measured under.
func ContractDifference(now, then Contract) string {
	var differing []string
	for _, part := range []struct{ name, now, then string }{
		{"the measure", now.Metric, then.Metric}, {"the origin", now.Origin, then.Origin},
		{"the suite", now.Suite, then.Suite}, {"the protocol", now.Protocol, then.Protocol},
		{"the comparison authority", now.Comparison.String(), then.Comparison.String()}, {"the recipe", now.Scope.String(), then.Scope.String()},
	} {
		if part.now != part.then {
			differing = append(differing, fmt.Sprintf("%s is %q, the reference's %q", part.name, part.now, part.then))
		}
	}
	return "measured on another contract: " + strings.Join(differing, "; ")
}

// holds reports how now holds the reference's measured cell then of a column,
// and why when it does not. It has regressed when now measured it on the same
// contract and direction and it got worse by more than the reference allows (see
// worse); now's own readings count the step being judged and allow nothing. A
// failed share regresses when more attempts failed than the earlier share predicts
// for the attempts counted now, by more than one. It is Unknown when now is not
// measured, measured only another contract, then has no contract, a serving cell's
// environment is not its reference's, as when only one of the two names any (two
// that both name none still compare), or the reference's earlier readings are too
// few to judge by.
func holds(column metric.Name, then, now Cell) (Verdict, string) {
	switch {
	case !now.Measured():
		return Unknown, "no current value: " + now.Reason
	case then.Contract == (Contract{}):
		return Unknown, "the reference cell was published without a contract"
	case (then.Contract.Origin == originEvaluation || then.Contract.Origin == originBenchmark) && !then.Contract.Comparison.Valid():
		return Unknown, then.Contract.Origin + " comparison authority is unknown"
	case now.Contract != then.Contract:
		return Unknown, ContractDifference(now.Contract, then.Contract)
	case then.Contract.Origin == originServing && now.Environment != then.Environment:
		return Unknown, fmt.Sprintf("measured under environment %q, the reference under %q", now.Environment, then.Environment)
	case now.Direction != then.Direction:
		return Unknown, fmt.Sprintf("improves in direction %q, the reference in %q", now.Direction, then.Direction)
	case column == metric.Failures && (then.Attempts == 0 || now.Attempts == 0):
		return Unknown, "the failed share was published without the attempts it counts"
	case then.unjudged != "":
		return Unknown, then.unjudged
	}
	declined := worse(cmp.Or(now.Direction, columnDirections[column]), now.Value, then.Value, then.allowed)
	if column == metric.Failures {
		declined = now.Value*float64(now.Attempts)-then.Value*float64(now.Attempts) > failureNoise
	}
	if declined {
		return regressed, fmt.Sprintf("%.9g to %.9g (%+.9g, %+.1f%%, allowed %.9g; %s)", then.Value, now.Value, now.Value-then.Value, percent*(now.Value-then.Value)/then.Value,
			max(then.allowed, unread(min(math.Abs(now.Value), math.Abs(then.Value)))), now.Label)
	}
	return holding, ""
}

// worse reports whether value is worse than reference in direction by more than
// allowed, or by more than the scorecard's printed digits show of the smaller of the
// two (unread), whichever is more. What is allowed is the reference's, measured before
// the value judged existed, so a value's own step never widens it. A value with no
// direction never is.
func worse(direction runrecord.Direction, value, reference, allowed float64) bool {
	advantage, directed := direction.Advantage(value, reference)
	return directed && -advantage > max(allowed, unread(min(math.Abs(value), math.Abs(reference))))
}

// printedDigits is how many significant digits the scorecard prints of a value
// (QualityText, and the table and the moves of the validate command).
const printedDigits = 4

// unread is half a unit of the last digit the scorecard prints of value: a move
// smaller than it is one no reader of the scorecard sees, and is no decline. It is
// what keeps a held-out quality computed from float32 logits in another order of
// summation, a re-measured perplexity that moves by parts in a billion, and a peak
// memory a few hundred bytes larger from reading as declines, while one case lost
// of ten thousand still is one. Zero has no digit and allows nothing.
func unread(value float64) float64 {
	if value == 0 {
		return 0
	}
	return half * math.Pow10(int(math.Floor(math.Log10(math.Abs(value))))-printedDigits+1)
}

// percent renders a change as users read it.
const percent = 100

// failureNoise is how many failures beyond the earlier share's prediction, at the
// attempts counted now, are chance: one more failed attempt is not a decline.
const failureNoise = 1

// The kinds of gap besides the regressions. Their names order the rows: trained
// models that fall below their base, then failures a user met, then declines,
// then served paths to compare with a reference output, then measurements to
// take, then held-out evaluations to define. A trained model below its base
// judges a training against its base model, not the record against its
// reference.
const (
	gapBelowBase  = "below-base"
	gapFailing    = "failing"
	gapUncompared = "uncompared"
	gapUnmeasured = "unmeasured"
	gapUnscored   = "unscored"
)

// gapKinds are the kinds of gap that name a task, each the middle of a gap's
// key.
var gapKinds = []string{gapBelowBase, gapFailing, gapUncompared, gapUnmeasured, gapUnscored}

// gapKeyPrefix begins every gap's key.
const gapKeyPrefix = "scorecard"

// gapKey names the gap of a kind on a task.
func gapKey(kind string, task recipe.Task) string {
	return strings.Join([]string{gapKeyPrefix, kind, string(task)}, "-")
}

// regressionKeyPrefix begins the key of a decline, which names the one cell that
// declined, by a digest of its model, task, measure and full contract.
const regressionKeyPrefix = gapKeyPrefix + "-regressed-"

// regressionDigits is how many hex digits of the digest the key keeps.
const regressionDigits = 16

// regressionKey names the decline of a cell: the model, the task, the measure
// and the full contract it was compared on, so a cell that stays declined is
// found again under the same key whatever value it declined to.
func regressionKey(row Row, column metric.Name, cell Cell) string {
	sum := sha256.Sum256(fmt.Appendf(nil, "%s|%s|%s|%+v", row.Model, row.Task, column, cell.Contract))
	return regressionKeyPrefix + hex.EncodeToString(sum[:])[:regressionDigits]
}

// KnownGapKey reports whether key names a gap the scorecard can propose: a
// kind of gap on a task the catalog holds, or the decline of a cell. A key that
// matches neither is not a gap that is closed; nothing could ever open it.
func KnownGapKey(key string) bool {
	if digest, found := strings.CutPrefix(key, regressionKeyPrefix); found {
		raw, err := hex.DecodeString(digest)
		return err == nil && hex.EncodedLen(len(raw)) == regressionDigits
	}
	return slices.ContainsFunc(gapKinds, func(kind string) bool {
		task, found := strings.CutPrefix(key, gapKey(kind, ""))
		return found && recipe.Task(task).Valid()
	})
}

// Restored reports whether the cell a decline's key names, read from rows, is
// back at target: measured, and not worse than it by more than the scorecard's
// printed digits show. A target already holds what the cell's readings allowed
// (Gap). found says whether a cell carries the key at all.
func Restored(rows []Row, key string, target float64) (restored, found bool) {
	for _, row := range rows {
		for _, column := range metric.Columns {
			for _, cell := range row.Measured(column) {
				if regressionKey(row, column, cell) == key {
					return !worse(cmp.Or(cell.Direction, columnDirections[column]), cell.Value, target, 0), true
				}
			}
		}
	}
	return false, false
}

// Gap is the work one task's shortfall on one metric asks for, with every
// row that falls short and its reason or values. Key names the gap the same
// way in every record, so its row is proposed once and its verifier finds
// it again. A decline names one cell, and Target is the value that cell is
// to be restored to: the worst value that would have held its reference, which
// is the reference's value, worse by what the cell's readings allowed. It is
// kept by the row proposed the first time and never replaced by a later, worse
// one.
type Gap struct {
	Key, Title, Detail string
	kind               string
	Target             float64
	Restore            bool
}

// bar is the worst value in direction that holds a reference's cell: its value,
// worse by what it allows.
func (c Cell) bar(direction runrecord.Direction) float64 {
	if direction == runrecord.DirectionMaximize {
		return c.Value - c.allowed
	}
	return c.Value + c.allowed
}

// Gaps lists the record's gaps against its reference, which may be nil: each
// cell no record measures, each failure-rate above zero, each cell that declined
// against its reference by more than the reference allows (one gap for each, by
// its model, task, measure and contract; a cell nothing compares, or that the
// record no longer has, is Unknown and reported by Moves, never filed), each
// held-out quality that is not applicable because its task's held-out
// evaluation is only proposed, and each trained model's held-out quality
// that fell below its base model's on the verdict that judged them.
func (record Record) Gaps(reference *Record) []Gap {
	var before Record
	if reference != nil {
		before = *reference
	}
	grouped := map[string]*Gap{}
	var declines []Gap
	// One gap per kind and task: its metrics are lines of the one row, so
	// a task's missing measurements are one piece of work.
	add := func(kind string, row *Row, column metric.Name, line string) {
		key := gapKey(kind, row.Task)
		if grouped[key] == nil {
			grouped[key] = &Gap{Key: key, Title: gapTitle(kind, row.Task), kind: kind}
		}
		grouped[key].Detail += "- " + row.Name + " " + string(column) + ": " + line + "\n"
	}
	for _, pair := range record.Pairs(before) {
		if pair.After == nil {
			continue
		}
		row := pair.After
		for _, column := range metric.Columns {
			for _, comparison := range pair.Compare(column) {
				if comparison.Verdict == regressed {
					target := comparison.Then.bar(cmp.Or(comparison.Now.Direction, columnDirections[column]))
					declines = append(declines, Gap{
						Key: regressionKey(*row, column, comparison.Now), kind: string(regressed), Restore: true, Target: target,
						Title:  fmt.Sprintf("Restore %s %s %s to %.9g %s", row.Name, row.Task, column, target, comparison.Then.Unit),
						Detail: fmt.Sprintf("- %s %s, %+v: %s\n", row.Name, column, comparison.Now.Contract, comparison.Why),
					})
				}
			}
			switch cell := row.Cells[column]; {
			case !cell.Measured() && !cell.NotApplicable:
				add(gapUnmeasured, row, column, cell.Reason)
			case column == metric.Quality && cell.NotApplicable && cell.Proposed:
				// Only a held-out evaluation still proposed is work to define; a quality
				// declared not applicable, with its reason, is no gap.
				add(gapUnscored, row, column, cell.Reason)
			case column == metric.Failures && cell.Value > 0:
				add(gapFailing, row, column, cell.Label)
			}
		}
		if cell := row.Cells[metric.Quality]; cell.Measured() && cell.Base != nil && worse(cell.Direction, cell.Value, *cell.Base, 0) {
			add(gapBelowBase, row, metric.Quality, cell.QualityText())
		}
		if row.PathUnchecked() {
			add(gapUncompared, row, metric.Quality, uncompared)
		}
	}
	gaps := declines
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
	case gapFailing:
		return fmt.Sprintf("Fix the %s attempts that failed", task)
	case gapUncompared:
		return fmt.Sprintf("Compare the path a %s request takes with a reference output for each model it is scored on", task)
	case gapUnscored:
		return fmt.Sprintf("Define the held-out evaluation that measures %s quality", task)
	default:
		return fmt.Sprintf("Measure %s's unmeasured cells", task)
	}
}
