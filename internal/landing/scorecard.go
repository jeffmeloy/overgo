package landing

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// sliceDirectory holds the scorecard slice documents: one fixed input per
// task, measured on that task's smallest local model. A change to one
// reaches the slices.
const sliceDirectory = "fixtures/scorecard_slice"

// inferenceSlice runs first, while the device holds no other model, and is
// the slice a bare timed metric in a prediction names.
const inferenceSlice = "inference"

// sliceTrailer is the commit trailer that carries a landing's slices, one
// line each; a later landing that measures the same identity compares
// against it.
const sliceTrailer = "Scorecard-Slice"

// trailerSeparator joins a commit's slice trailers on one log line; JSON
// always escapes it, so no record contains it.
const trailerSeparator = "\x1f"

// bytesPerMiB is the unit peak device memory is compared in.
const bytesPerMiB = 1 << 20

// pairedRounds is how many A B B A rounds of HEAD's benchmark and the
// tree's a predicted timed metric runs, about two minutes on the smallest
// model. Simulated at 1% run-to-run noise, four rounds refuse an unchanged
// tree about once in 100,000 trials per metric and a 5% worsening 96% of
// the time; two rounds refused an unchanged tree once in 180.
const pairedRounds = 4

// percent renders a relative change as users read it.
const percent = 100

// The two sides of a paired run: HEAD's benchmark and the tree's.
const (
	headSide = iota
	treeSide
	sides
)

// sideNames names each side in a detail.
var sideNames = [sides]string{headSide: "HEAD", treeSide: "the tree"}

// abba is one paired round's order, which cancels a drift across it.
var abba = []int{headSide, treeSide, treeSide, headSide}

// slice is what the scorecard slices depend on: the benchmark binary's
// closure, which embeds the built kernels, the module versions and the
// documents.
var slice = lane{name: "scorecard", packages: []string{"./cmd/benchmark"},
	prefixes: []string{"go.mod", "go.sum", sliceDirectory + "/"}}

// timedMetric is one of a slice's timed cells, whether a user feels it
// better when it falls, and the unit it is read in.
type timedMetric struct {
	name          metric.Name
	lowerIsBetter bool
	unit          string
}

// timedMetrics are the timed cells: a shorter wait for the first output, a
// faster stream, a quicker load.
var timedMetrics = []timedMetric{{metric.FirstOutput, true, "ms"}, {metric.Throughput, false, "tok/s"}, {metric.Load, true, "ms"}}

// baseline is what earlier landings recorded of one identity: the newest
// outcome, measured or failed, and the newest measurement, which the exact
// cells compare with even when a failure landed after it.
type baseline struct {
	latest, measured             metric.Record
	latestCommit, measuredCommit string
}

// scorecard runs the scorecard's slices when the change reaches what they
// measure: each document's task on its smallest local model, all in one
// benchmark process. A slice's exact cells -- the outputs and peak device
// memory -- are compared with the newest earlier landing that measured the
// same model, recipe, workload, options and environment. A timed cell is
// reported unpaired, since one run on each side cannot tell a change from
// noise, unless the message's predicted effect names it: naming one claims
// it does not get worse, so HEAD's benchmark runs that slice beside the
// tree's and a paired worsening beyond the noise of same-binary runs
// refuses the landing. A slice that does not measure says why: unavailable
// when the device is claimed or out of memory; failed otherwise, which
// refuses the landing only when HEAD's benchmark measures what the tree
// failed to.
func (l *landing) scorecard(ctx context.Context) error {
	reached, err := l.reaches(ctx, slice)
	if err != nil {
		return err
	}
	documents, err := sliceDocuments(l.Repo)
	if err != nil {
		return err
	}
	if !reached || len(documents) == 0 {
		l.details["scorecard"] = "not reached"
		return nil
	}
	directory, err := os.MkdirTemp("", "overgo-scorecard-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(directory)
	tree, err := l.buildBenchmark(ctx, directory, "tree", "")
	if err != nil {
		l.details["scorecard"] = "failed: the benchmark does not build"
		return nil
	}
	records, err := l.runSlices(ctx, directory, tree, documents)
	if err != nil {
		return err
	}
	head := l.headBinary(ctx, directory)
	notes, err := l.blame(ctx, directory, head, documents, records)
	if err != nil {
		return err
	}
	baselines, err := l.sliceBaselines(ctx)
	if err != nil {
		return err
	}
	names := make([]string, len(documents))
	for index, document := range documents {
		names[index] = sliceName(document)
	}
	predicted, absent := predictedTimed(l.Message, names)
	var parts []string
	for _, task := range absent {
		parts = append(parts, "not paired: no "+task+" slice")
	}
	failed, counted := 0, 0
	for index, record := range records {
		part := describeSlice(record, baselines)
		if note := notes[record.Slice]; note != "" {
			part += "; " + note
		}
		if metrics := predicted[record.Slice]; len(metrics) != 0 {
			paired, err := l.pair(ctx, directory, tree, head, documents[index], record, metrics)
			if err != nil {
				return err
			}
			part += "; " + paired
		}
		parts = append(parts, part)
		if record.Failure == "" && !measured(record) {
			continue
		}
		counted++
		if record.Failure != "" {
			failed++
		}
		encoded, err := json.Marshal(record)
		if err != nil {
			return err
		}
		l.trailers = append(l.trailers, sliceTrailer+": "+string(encoded))
	}
	l.details["scorecard"] = strings.Join(append(parts, fmt.Sprintf("%s %d of %d slices", metric.Failures, failed, counted)), " | ")
	return nil
}

// sliceDocuments are the repository's slice documents, the inference slice
// first, then by name.
func sliceDocuments(repo string) ([]string, error) {
	matches, err := filepath.Glob(filepath.Join(repo, filepath.FromSlash(sliceDirectory), "*.json"))
	if err != nil {
		return nil, err
	}
	documents := make([]string, len(matches))
	for index, match := range matches {
		documents[index] = path.Join(sliceDirectory, filepath.Base(match))
	}
	if at := slices.IndexFunc(documents, func(document string) bool { return sliceName(document) == inferenceSlice }); at > 0 {
		first := documents[at]
		documents = append([]string{first}, slices.Delete(documents, at, at+1)...)
	}
	return documents, nil
}

// sliceName is the slice a document measures: its file name.
func sliceName(document string) string {
	return strings.TrimSuffix(path.Base(document), path.Ext(document))
}

// measured reports whether a record is a measurement.
func measured(record metric.Record) bool {
	return record.Failure == "" && record.Unmeasured == "" && !record.Selected()
}

// buildBenchmark builds cmd/benchmark into the directory under name, through
// the overlay when one is given.
func (l *landing) buildBenchmark(ctx context.Context, directory, name, overlay string) (string, error) {
	binary := filepath.Join(directory, name)
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	arguments := []string{"build", "-o", binary}
	if overlay != "" {
		arguments = append(arguments, "-overlay", overlay)
	}
	_, err := l.run(ctx, nil, "go", append(arguments, "./cmd/benchmark")...)
	return binary, err
}

// headBinary builds HEAD's benchmark beside the tree the first time it is
// asked for, and returns the same binary after: an overlay maps each path
// the change touched to HEAD's bytes, or to nothing where HEAD lacks it, so
// the packages the change left alone stay cached.
func (l *landing) headBinary(ctx context.Context, directory string) func() (string, error) {
	return sync.OnceValues(func() (string, error) { return l.buildHead(ctx, directory) })
}

func (l *landing) buildHead(ctx context.Context, directory string) (string, error) {
	replace := map[string]string{}
	for index, name := range l.paths {
		absolute := filepath.Join(l.Repo, filepath.FromSlash(name))
		data, err := gitauthority.Query(ctx, l.Repo, "show", "HEAD:"+name)
		if err != nil {
			replace[absolute] = ""
			continue
		}
		held := filepath.Join(directory, "head-"+strconv.Itoa(index))
		if err := os.WriteFile(held, data, clioptions.PrivateFileMode); err != nil {
			return "", err
		}
		replace[absolute] = held
	}
	encoded, err := json.Marshal(map[string]any{"Replace": replace})
	if err != nil {
		return "", err
	}
	overlay := filepath.Join(directory, "head-overlay.json")
	if err := os.WriteFile(overlay, encoded, clioptions.PrivateFileMode); err != nil {
		return "", err
	}
	return l.buildBenchmark(ctx, directory, "head", overlay)
}

// runSlices runs the documents through one benchmark process and returns
// one outcome per document, in order. The process prints what each slice
// selected before it measures; when it crashes, the first document without
// an outcome is the one it crashed on, keeping what it selected, and the
// documents after it run in a process of their own.
func (l *landing) runSlices(ctx context.Context, directory, binary string, documents []string) ([]metric.Record, error) {
	var records []metric.Record
	for len(documents) != 0 {
		printed, ended, err := l.invokeSlices(ctx, directory, binary, documents)
		if err != nil {
			return nil, err
		}
		outcomes, selections := map[string]metric.Record{}, map[string]metric.Record{}
		for _, record := range printed {
			if record.Selected() {
				selections[record.Slice] = record
			} else {
				outcomes[record.Slice] = record
			}
		}
		next := len(documents)
		for index, document := range documents {
			name := sliceName(document)
			if outcome, found := outcomes[name]; found {
				records = append(records, outcome)
				continue
			}
			crashed := ended(selections[name])
			crashed.Slice = name
			records = append(records, crashed)
			next = index + 1
			break
		}
		documents = documents[next:]
	}
	return records, nil
}

// invokeSlices runs one benchmark process over the documents and returns
// the records it printed and how a document it printed no outcome for
// ended: failed, with the process's own error. The runner prints a claimed
// or exhausted device as an unavailable record, so no process exit stands
// for one.
func (l *landing) invokeSlices(ctx context.Context, directory, binary string, documents []string) ([]metric.Record, func(metric.Record) metric.Record, error) {
	errorFile, err := os.CreateTemp(directory, "error-*.txt")
	if err != nil {
		return nil, nil, err
	}
	if err := errorFile.Close(); err != nil {
		return nil, nil, err
	}
	var output, diagnostics bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: binary, Args: append([]string{"-slice"}, documents...), Dir: l.Repo,
		Env:    append(os.Environ(), clioptions.ErrorFileEnvironment+"="+errorFile.Name()),
		Stdout: &output, Stderr: &diagnostics,
	})
	if err != nil {
		return nil, nil, err
	}
	var printed []metric.Record
	for line := range strings.Lines(output.String()) {
		var record metric.Record
		if json.Unmarshal([]byte(line), &record) == nil && record.Slice != "" {
			printed = append(printed, record)
		}
	}
	ended := func(selected metric.Record) metric.Record {
		selected.Failure = "the benchmark printed no outcome"
		if receipt.ExitCode != 0 {
			selected.Failure = fmt.Sprintf("exit %d: %s", receipt.ExitCode, failureCause(errorFile.Name(), diagnostics.String()))
		}
		selected.Metrics = map[metric.Name]float64{metric.Failures: 1}
		return selected
	}
	return printed, ended, nil
}

// failureCause is the benchmark's own error as it wrote it, else the panic
// line of its standard error, else that stream's last line: a panic ends
// with a stack frame, not its cause.
func failureCause(errorFile, diagnostics string) string {
	if data, err := os.ReadFile(errorFile); err == nil && len(bytes.TrimSpace(data)) != 0 {
		return strings.Join(strings.Fields(string(data)), " ")
	}
	lines := strings.Split(strings.TrimSpace(diagnostics), "\n")
	if at := slices.IndexFunc(lines, func(line string) bool { return strings.HasPrefix(line, "panic: ") }); at >= 0 {
		return strings.TrimSpace(lines[at])
	}
	return strings.TrimSpace(lines[len(lines)-1])
}

// blame refuses a landing whose slice fails on the tree where HEAD's
// benchmark measures the same identity, or where the tree fails before
// selecting a model and HEAD's selects one and measures it: the change
// broke that path. HEAD runs only the failed documents. A failure HEAD
// shares or cannot run refuses nothing, and HEAD measuring another
// identity is noted as not compared.
func (l *landing) blame(ctx context.Context, directory string, head func() (string, error), documents []string, records []metric.Record) (map[string]string, error) {
	var failed []string
	for index, record := range records {
		if record.Failure != "" {
			failed = append(failed, documents[index])
		}
	}
	if len(failed) == 0 {
		return nil, nil
	}
	binary, err := head()
	if err != nil {
		return nil, nil
	}
	headRecords, err := l.runSlices(ctx, directory, binary, failed)
	if err != nil {
		return nil, err
	}
	notes := map[string]string{}
	var refusals []string
	for _, tree := range records {
		at := slices.IndexFunc(headRecords, func(record metric.Record) bool { return record.Slice == tree.Slice })
		if tree.Failure == "" || at < 0 || !measured(headRecords[at]) {
			continue
		}
		switch headRecord := headRecords[at]; {
		case tree.Identity == "":
			refusals = append(refusals, fmt.Sprintf("the %s slice selects %s and measures it on HEAD but fails on the tree before selecting a model (%s)",
				tree.Slice, headRecord.Model, tree.Failure))
		case tree.Identity == headRecord.Identity:
			refusals = append(refusals, treeFailure(tree).Error())
		default:
			notes[tree.Slice] = "not compared: HEAD measures " + headRecord.Model + " under another identity"
		}
	}
	if len(refusals) != 0 {
		return nil, errors.New(strings.Join(refusals, "; "))
	}
	return notes, nil
}

// treeFailure refuses a landing whose slice fails on the tree where HEAD's
// benchmark measures the same identity: the change broke it.
func treeFailure(record metric.Record) error {
	return fmt.Errorf("the %s slice measures on HEAD and fails on the tree (%s)", record.Slice, record.Failure)
}

// predictedTimed is, per slice, the timed metrics the message's predicted
// effect names by their exact names: a metric right after a task's name
// belongs to that task's slice (speech first-output), and a bare one to the
// inference slice. A task named that has no slice document is returned
// apart, so its prediction reads as unpaired rather than as inference's.
func predictedTimed(message []byte, names []string) (map[string][]timedMetric, []string) {
	predicted, found := section(message, predictedSection)
	if !found {
		return nil, nil
	}
	named := map[string][]timedMetric{}
	var absent []string
	for _, timed := range timedMetrics {
		for rest := predicted; ; {
			at := strings.Index(rest, string(timed.name))
			if at < 0 {
				break
			}
			slice := inferenceSlice
			if before := strings.Fields(rest[:at]); len(before) != 0 && strings.HasSuffix(rest[:at], " ") {
				if task := strings.TrimFunc(before[len(before)-1], func(r rune) bool { return !unicode.IsLetter(r) && r != '-' }); recipe.Task(task).Valid() {
					slice = task
				}
			}
			rest = rest[at+len(timed.name):]
			switch {
			case !slices.Contains(names, slice):
				if !slices.Contains(absent, slice) {
					absent = append(absent, slice)
				}
			case !slices.Contains(named[slice], timed):
				named[slice] = append(named[slice], timed)
			}
		}
	}
	return named, absent
}

// pair runs one slice on HEAD's benchmark and the tree's in A B B A rounds
// and reports each predicted metric's change between the sides' means
// against its noise. A predicted metric that worsened beyond its noise
// refuses the landing, as does a tree run that fails where HEAD's measured
// the same identity; a run that is unavailable, a HEAD run that fails, a
// run under another identity, or a metric the slice does not measure leaves
// the metrics unpaired.
func (l *landing) pair(ctx context.Context, directory, tree string, head func() (string, error), document string, record metric.Record, predicted []timedMetric) (string, error) {
	if !measured(record) {
		return "not paired: " + record.Slice + " did not measure", nil
	}
	var absent []string
	for _, timed := range predicted {
		if _, found := record.Metrics[timed.name]; !found {
			absent = append(absent, string(timed.name))
		}
	}
	if len(absent) != 0 {
		return "not paired: " + record.Slice + " measures no " + strings.Join(absent, ", "), nil
	}
	headBinary, err := head()
	if err != nil {
		return "not paired: HEAD's benchmark does not build", nil
	}
	binaries := [sides]string{headSide: headBinary, treeSide: tree}
	var runs [sides][]metric.Record
	for range pairedRounds {
		for _, side := range abba {
			results, err := l.runSlices(ctx, directory, binaries[side], []string{document})
			if err != nil {
				return "", err
			}
			run := results[0]
			if !measured(run) {
				// Each round runs HEAD first, so HEAD has measured this
				// identity before any tree run is judged.
				if side == treeSide && run.Failure != "" && run.Identity == record.Identity {
					return "", treeFailure(run)
				}
				return "not paired: " + sideNames[side] + " " + cmp.Or(run.Unmeasured, "failed: "+run.Failure), nil
			}
			if run.Identity != record.Identity {
				return fmt.Sprintf("not paired: %s measures %s under another identity", sideNames[side], run.Model), nil
			}
			runs[side] = append(runs[side], run)
		}
	}
	var cells, worse []string
	for _, timed := range predicted {
		change, noise := pairedChange(runs, timed.name)
		cell := fmt.Sprintf("%s %+.1f%% (noise %.1f%%)", timed.name, change*percent, noise*percent)
		cells = append(cells, cell)
		if (change > 0) == timed.lowerIsBetter && math.Abs(change) > noise {
			worse = append(worse, cell)
		}
	}
	if len(worse) != 0 {
		return "", fmt.Errorf("the predicted effect names %s %s, which paired runs of HEAD and the tree measure worse beyond the noise of same-binary runs",
			record.Slice, strings.Join(worse, ", "))
	}
	return fmt.Sprintf("paired over %d rounds: %s", pairedRounds, strings.Join(cells, ", ")), nil
}

// pairedChange is the tree's relative change in the metric against HEAD,
// between the means of each side's runs, and the noise: the wider side's
// range over all its runs, relative to its mean. A slowdown that lands on
// one side's runs of one round widens that side's range with it.
func pairedChange(runs [sides][]metric.Record, name metric.Name) (float64, float64) {
	var means [sides]float64
	var noise float64
	for side, records := range runs {
		values := make([]float64, len(records))
		for index, record := range records {
			values[index] = record.Metrics[name]
			means[side] += values[index] / float64(len(records))
		}
		noise = max(noise, (slices.Max(values)-slices.Min(values))/means[side])
	}
	return means[treeSide]/means[headSide] - 1, noise
}

// sliceBaselines indexes, by identity, the newest first-parent landing's
// outcome and the newest one that measured, with their commits. A merged
// branch's landings are not the mainline's baselines, and a record from
// before slices were named is found by its identity like any other.
func (l *landing) sliceBaselines(ctx context.Context) (map[string]baseline, error) {
	log, err := gitauthority.Query(ctx, l.Repo, "log", "--first-parent", "--grep=^"+sliceTrailer+": ",
		"--format=%h %(trailers:key="+sliceTrailer+",valueonly,separator=%x1f)")
	if err != nil {
		return nil, err
	}
	index := map[string]baseline{}
	for line := range strings.Lines(string(log)) {
		commit, values, _ := strings.Cut(strings.TrimSpace(line), " ")
		for value := range strings.SplitSeq(values, trailerSeparator) {
			var record metric.Record
			if json.Unmarshal([]byte(value), &record) != nil || record.Identity == "" {
				continue
			}
			entry := index[record.Identity]
			if entry.latestCommit == "" {
				entry.latest, entry.latestCommit = record, commit
			}
			if entry.measuredCommit == "" && measured(record) {
				entry.measured, entry.measuredCommit = record, commit
			}
			index[record.Identity] = entry
		}
	}
	return index, nil
}

// describeSlice says what one slice measured: a failure against the newest
// outcome of its identity, a measurement's exact cells against the newest
// measurement of it, then its timed cells as measured.
func describeSlice(record metric.Record, baselines map[string]baseline) string {
	prefix := record.Slice + ": "
	if record.Unmeasured != "" {
		return prefix + record.Unmeasured
	}
	var base baseline
	if record.Identity != "" {
		base = baselines[record.Identity]
	}
	if record.Failure != "" {
		switch {
		case record.Identity == "":
			return prefix + "fails before selecting a model: " + record.Failure
		case base.latestCommit == "":
			return prefix + record.Model + " fails: " + record.Failure
		case base.latest.Failure != "":
			return fmt.Sprintf("%s%s still fails (as at %s): %s", prefix, record.Model, base.latestCommit, record.Failure)
		default:
			return fmt.Sprintf("%s%s fails (measured at %s): %s", prefix, record.Model, base.latestCommit, record.Failure)
		}
	}
	compared := base.measuredCommit != ""
	var outputs string
	switch {
	case record.Outputs == metric.VariedOutputs:
		outputs = "outputs vary between runs"
	case !compared:
		outputs = "outputs recorded"
	case base.measured.Outputs == metric.VariedOutputs:
		outputs = "outputs not compared: the baseline's varied"
	case record.Outputs == base.measured.Outputs:
		outputs = "outputs identical"
	default:
		outputs = "outputs differ"
	}
	if peak, found := record.Metrics[metric.PeakMemory]; found {
		outputs += fmt.Sprintf("; %s %.1f MiB", metric.PeakMemory, peak/bytesPerMiB)
		if basePeak, found := base.measured.Metrics[metric.PeakMemory]; compared && found {
			outputs += fmt.Sprintf(" (%+.1f)", (peak-basePeak)/bytesPerMiB)
		}
	}
	exact := "no earlier slice of this identity: " + outputs
	if compared {
		exact = fmt.Sprintf("against %s: %s", base.measuredCommit, outputs)
	}
	if base.latest.Failure != "" {
		exact = fmt.Sprintf("measures (failed at %s); %s", base.latestCommit, exact)
	}
	var timed []string
	for _, cell := range timedMetrics {
		if value, found := record.Metrics[cell.name]; found {
			timed = append(timed, fmt.Sprintf("%s %.0f %s", cell.name, value, cell.unit))
		}
	}
	if len(timed) == 0 {
		return prefix + record.Model + ", " + exact
	}
	return prefix + record.Model + ", " + exact + "; unpaired " + strings.Join(timed, ", ")
}
