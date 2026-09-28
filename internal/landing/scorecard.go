package landing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/processcontrol"
	"overgo/internal/scorecard/metric"
)

// sliceDocument is the scorecard slice a landing measures: one fixed
// inference input on the smallest local text model. The other tasks' slices
// and paired timing belong to the scheduled scorecard.
const sliceDocument = "fixtures/scorecard_slice/inference.json"

// sliceTrailer is the commit trailer that carries a landing's slice; a later
// landing that measures the same identity compares against it.
const sliceTrailer = "Scorecard-Slice"

// trailerSeparator joins a commit's slice trailers on one log line; JSON
// always escapes it, so no record contains it.
const trailerSeparator = "\x1f"

// bytesPerMiB is the unit peak device memory is compared in.
const bytesPerMiB = 1 << 20

// slice is what the scorecard slice depends on: the benchmark binary's
// closure, which embeds the built kernels, the module versions and the
// document.
var slice = lane{packages: []string{"./cmd/benchmark"}, prefixes: []string{"go.mod", "go.sum", sliceDocument}}

// timedMetrics are the timed cells, reported as measured: one run cannot
// tell a change from noise.
var timedMetrics = []struct {
	name metric.Name
	unit string
}{{metric.FirstOutput, "ms"}, {metric.Throughput, "tok/s"}, {metric.Load, "ms"}}

// scorecard runs the slice when the change reaches what it measures. Its
// exact cells -- the outputs and peak device memory -- are compared with the
// newest earlier landing that measured the same model, recipe, workload,
// options and environment. It reports and never refuses: a slice that does
// not measure says why.
func (l *landing) scorecard(ctx context.Context) error {
	reached, err := l.reaches(ctx, slice)
	if err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(l.Repo, filepath.FromSlash(sliceDocument))); !reached || err != nil {
		l.details["scorecard"] = "not reached"
		return nil
	}
	directory, err := os.MkdirTemp("", "overgo-scorecard-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(directory)
	binary := filepath.Join(directory, "benchmark")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if _, err := l.run(ctx, nil, "go", "build", "-o", binary, "./cmd/benchmark"); err != nil {
		l.details["scorecard"] = "failed: the benchmark does not build"
		return nil
	}
	record, err := l.runSlice(ctx, directory, binary)
	if err != nil {
		return err
	}
	base, commit, err := l.sliceBaseline(ctx, record.Identity)
	if err != nil {
		return err
	}
	l.details["scorecard"] = describeSlice(record, base, commit)
	if record.Failure != "" || measured(record) {
		encoded, err := json.Marshal(record)
		if err != nil {
			return err
		}
		l.trailers = append(l.trailers, sliceTrailer+": "+string(encoded))
	}
	return nil
}

// measured reports whether a record is a measurement.
func measured(record metric.Record) bool {
	return record.Failure == "" && record.Unmeasured == "" && !record.Selected()
}

// runSlice runs the benchmark over the slice and returns its outcome. The
// benchmark prints what it selected before it measures, so a run that ends
// without an outcome keeps the model it selected and fails with the
// benchmark's own error. A claimed or exhausted device is printed as an
// unavailable record, so no exit stands for one.
func (l *landing) runSlice(ctx context.Context, directory, binary string) (metric.Record, error) {
	errorFile, err := os.CreateTemp(directory, "error-*.txt")
	if err != nil {
		return metric.Record{}, err
	}
	if err := errorFile.Close(); err != nil {
		return metric.Record{}, err
	}
	var output, diagnostics bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: binary, Args: []string{"-slice", sliceDocument}, Dir: l.Repo,
		Env:    append(os.Environ(), clioptions.ErrorFileEnvironment+"="+errorFile.Name()),
		Stdout: &output, Stderr: &diagnostics,
	})
	if err != nil {
		return metric.Record{}, err
	}
	var selected metric.Record
	for line := range strings.Lines(output.String()) {
		var record metric.Record
		switch {
		case json.Unmarshal([]byte(line), &record) != nil || record.Slice == "":
		case record.Selected():
			selected = record
		default:
			return record, nil
		}
	}
	selected.Failure = "the benchmark printed no outcome"
	if receipt.ExitCode != 0 {
		selected.Failure = fmt.Sprintf("exit %d: %s", receipt.ExitCode, failureCause(errorFile.Name(), diagnostics.String()))
	}
	selected.Metrics = map[metric.Name]float64{metric.Failures: 1}
	return selected, nil
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

// sliceBaseline is the newest first-parent landing that measured the
// identity, and its commit. A merged branch's landings are not the
// mainline's baselines.
func (l *landing) sliceBaseline(ctx context.Context, identity string) (metric.Record, string, error) {
	if identity == "" {
		return metric.Record{}, "", nil
	}
	log, err := gitauthority.Query(ctx, l.Repo, "log", "--first-parent", "--grep=^"+sliceTrailer+": ",
		"--format=%h %(trailers:key="+sliceTrailer+",valueonly,separator=%x1f)")
	if err != nil {
		return metric.Record{}, "", err
	}
	for line := range strings.Lines(string(log)) {
		commit, values, _ := strings.Cut(strings.TrimSpace(line), " ")
		for value := range strings.SplitSeq(values, trailerSeparator) {
			var record metric.Record
			if json.Unmarshal([]byte(value), &record) == nil && record.Identity == identity && measured(record) {
				return record, commit, nil
			}
		}
	}
	return metric.Record{}, "", nil
}

// describeSlice says what the slice measured: a failure, or a measurement's
// exact cells against the newest measurement of its identity, then its timed
// cells as measured.
func describeSlice(record metric.Record, base metric.Record, commit string) string {
	prefix := record.Slice + ": "
	switch {
	case record.Unmeasured != "":
		return prefix + record.Unmeasured
	case record.Failure != "" && record.Identity == "":
		return prefix + "fails before selecting a model: " + record.Failure
	case record.Failure != "" && commit != "":
		return fmt.Sprintf("%s%s fails (measured at %s): %s", prefix, record.Model, commit, record.Failure)
	case record.Failure != "":
		return prefix + record.Model + " fails: " + record.Failure
	}
	var outputs string
	switch {
	case record.Outputs == metric.VariedOutputs:
		outputs = "outputs vary between runs"
	case commit == "":
		outputs = "outputs recorded"
	case base.Outputs == metric.VariedOutputs:
		outputs = "outputs not compared: the baseline's varied"
	case record.Outputs == base.Outputs:
		outputs = "outputs identical"
	default:
		outputs = "outputs differ"
	}
	if peak, found := record.Metrics[metric.PeakMemory]; found {
		outputs += fmt.Sprintf("; %s %.1f MiB", metric.PeakMemory, peak/bytesPerMiB)
		if basePeak, found := base.Metrics[metric.PeakMemory]; commit != "" && found {
			outputs += fmt.Sprintf(" (%+.1f)", (peak-basePeak)/bytesPerMiB)
		}
	}
	exact := "no earlier slice of this identity: " + outputs
	if commit != "" {
		exact = fmt.Sprintf("against %s: %s", commit, outputs)
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
	return prefix + record.Model + ", " + exact + "; " + strings.Join(timed, ", ")
}
