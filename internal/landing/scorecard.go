package landing

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/processcontrol"
	"overgo/internal/scorecard/metric"
)

// sliceWorkload is the one prompt every slice runs: the leading MMLU-Pro
// conversations of adaptive_new's SFT content, concatenated chat-style, so
// the slice reads a long prompt and answers it.
const sliceWorkload = "fixtures/scorecard_slice_prompts.json"

// sliceTrailer is the commit trailer that carries a landing's slice; a later
// landing that measures the same identity compares against it.
const sliceTrailer = "Scorecard-Slice"

// bytesPerMiB is the unit peak device memory is compared in.
const bytesPerMiB = 1 << 20

// variedOutputs stands for the outputs of a slice whose runs disagreed:
// there is no one output to compare.
const variedOutputs = "varies"

// pairedRounds is how many A B B A rounds of HEAD's benchmark and the
// tree's a predicted timed metric runs, about two minutes on the smallest
// model. Simulated at 1% run-to-run noise, four rounds refuse an unchanged
// tree about once in 100,000 trials per metric and a 5% worsening 96% of
// the time; two rounds refused an unchanged tree once in 180.
const pairedRounds = 4

// percent renders a relative change as users read it.
const percent = 100

// sliceFailed opens the reason of a slice run that failed rather than found
// its device claimed or out of memory.
const sliceFailed = "failed: "

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

// slice is what the scorecard slice depends on: the benchmark binary's
// closure, which embeds the built kernels, the module versions and the
// workload.
var slice = lane{name: "scorecard", packages: []string{"./cmd/benchmark"},
	prefixes: []string{"go.mod", "go.sum", sliceWorkload}}

// timedMetric is one of the slice's timed cells and whether a user feels it
// better when it falls.
type timedMetric struct {
	name          metric.Name
	lowerIsBetter bool
}

// timedMetrics are the slice's timed cells: a shorter wait for the first
// output, a faster stream, a quicker load.
var timedMetrics = []timedMetric{{metric.FirstOutput, true}, {metric.Throughput, false}, {metric.Load, true}}

// sliceRecord is one landing's slice as its trailer carries it: the
// identity it was measured under, the model's name, the generated outputs,
// and each measured metric by its scorecard name.
type sliceRecord struct {
	Identity string                  `json:"identity"`
	Model    string                  `json:"model"`
	Outputs  string                  `json:"outputs"`
	Metrics  map[metric.Name]float64 `json:"metrics"`
}

// scorecard runs the scorecard's cheap slice when the change reaches what
// it measures: the smallest local model on one fixed prompt. The exact
// cells -- the outputs and peak device memory -- are compared with the
// newest earlier landing that measured the same model, recipe, workload,
// options and environment. A timed cell is reported unpaired, since one run
// on each side cannot tell a change from noise, unless the message's
// predicted effect names it: naming one claims it does not get worse, so
// HEAD's benchmark runs beside the tree's and a paired worsening beyond the
// noise of same-binary runs refuses the landing. A run that does not
// measure says why: unavailable when the device is claimed or out of
// memory; failed otherwise, which refuses the landing only when HEAD's
// benchmark measures.
func (l *landing) scorecard(ctx context.Context) error {
	reached, err := l.reaches(ctx, slice)
	if err != nil {
		return err
	}
	if !reached {
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
		l.details["scorecard"] = sliceFailed + "the benchmark does not build"
		return nil
	}
	record, missing, err := l.runSlice(ctx, directory, tree)
	if err != nil {
		return err
	}
	if missing != "" {
		l.details["scorecard"] = missing
		if !strings.HasPrefix(missing, sliceFailed) {
			return nil
		}
		head, err := l.buildHead(ctx, directory)
		if err != nil {
			return nil
		}
		if _, headMissing, err := l.runSlice(ctx, directory, head); err != nil || headMissing != "" {
			return err
		}
		return treeFailure(missing)
	}
	baseline, commit, err := l.sliceBaseline(ctx, record.Identity)
	if err != nil {
		return err
	}
	detail := describeSlice(record, baseline, commit)
	if predicted := predictedTimed(l.Message); len(predicted) != 0 {
		paired, err := l.pair(ctx, directory, tree, record.Identity, predicted)
		if err != nil {
			return err
		}
		detail += "; " + paired
	}
	l.details["scorecard"] = detail
	encoded, err := json.Marshal(record)
	if err != nil {
		return err
	}
	l.trailer = sliceTrailer + ": " + string(encoded)
	return nil
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

// buildHead builds HEAD's benchmark beside the tree: an overlay maps each
// path the change touched to HEAD's bytes, or to nothing where HEAD lacks
// it, so the packages the change left alone stay cached.
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

// runSlice runs a built benchmark once on the slice, so its exit status is
// the benchmark's own. A run that does not measure returns why, in one
// line; each run names its own error file, so no run reads another's.
func (l *landing) runSlice(ctx context.Context, directory, binary string) (sliceRecord, string, error) {
	errorFile, err := os.CreateTemp(directory, "error-*.txt")
	if err != nil {
		return sliceRecord{}, "", err
	}
	if err := errorFile.Close(); err != nil {
		return sliceRecord{}, "", err
	}
	var output, diagnostics bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: binary, Args: []string{"-smallest", "-prompt-suite", sliceWorkload}, Dir: l.Repo,
		Env:    append(os.Environ(), clioptions.ErrorFileEnvironment+"="+errorFile.Name()),
		Stdout: &output, Stderr: &diagnostics,
	})
	if err != nil {
		return sliceRecord{}, "", err
	}
	switch receipt.ExitCode {
	case 0:
	case processcontrol.ResourceBusyExitCode, processcontrol.DeviceMemoryExitCode:
		return sliceRecord{}, "unavailable: " + failureCause(errorFile.Name(), diagnostics.String()), nil
	default:
		return sliceRecord{}, fmt.Sprintf("%sexit %d: %s", sliceFailed, receipt.ExitCode, failureCause(errorFile.Name(), diagnostics.String())), nil
	}
	var measured struct {
		Model       string  `json:"model_id"`
		Recipe      string  `json:"recipe_id"`
		Workload    string  `json:"workload_digest"`
		Options     string  `json:"options_digest"`
		Environment string  `json:"environment_id"`
		Name        string  `json:"model_name"`
		LoadMS      float64 `json:"load_ms"`
		PeakBytes   uint64  `json:"device_peak_bytes"`
		Summary     struct {
			FirstOutputMS   float64 `json:"ttft_ms_p50"`
			TokensPerSecond float64 `json:"decode_tokens_per_second_p50"`
		} `json:"summary"`
		Runs []struct {
			Outputs []string `json:"output_digests"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(output.Bytes(), &measured); err != nil || len(measured.Runs) == 0 {
		return sliceRecord{}, sliceFailed + "the benchmark printed no measured runs", nil
	}
	identity := sha256.Sum256([]byte(strings.Join([]string{
		measured.Model, measured.Recipe, measured.Workload, measured.Options, measured.Environment,
	}, "\n")))
	outputs := strings.Join(measured.Runs[0].Outputs, ",")
	for _, run := range measured.Runs {
		if !slices.Equal(run.Outputs, measured.Runs[0].Outputs) {
			outputs = variedOutputs
		}
	}
	return sliceRecord{
		Identity: hex.EncodeToString(identity[:]), Model: measured.Name, Outputs: outputs,
		Metrics: map[metric.Name]float64{
			metric.PeakMemory:  float64(measured.PeakBytes),
			metric.FirstOutput: measured.Summary.FirstOutputMS,
			metric.Throughput:  measured.Summary.TokensPerSecond,
			metric.Load:        measured.LoadMS,
		},
	}, "", nil
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

// treeFailure refuses a landing whose slice fails on the tree after HEAD's
// measured: the change broke it.
func treeFailure(missing string) error {
	return fmt.Errorf("the slice measures on HEAD and fails on the tree (%s)", missing)
}

// predictedTimed is the timed metrics the message's predicted effect names
// by their exact names.
func predictedTimed(message []byte) []timedMetric {
	predicted, found := section(message, predictedSection)
	if !found {
		return nil
	}
	var named []timedMetric
	for _, timed := range timedMetrics {
		if strings.Contains(predicted, string(timed.name)) {
			named = append(named, timed)
		}
	}
	return named
}

// pair runs HEAD's benchmark and the tree's in A B B A rounds and reports
// each predicted metric's change between the sides' means against its
// noise. A predicted metric that worsened beyond its noise refuses the
// landing, as does a tree run that fails after HEAD's measured; a run that
// is unavailable, a HEAD run that fails, or a run under another identity
// leaves the metrics unpaired.
func (l *landing) pair(ctx context.Context, directory, tree, identity string, predicted []timedMetric) (string, error) {
	head, err := l.buildHead(ctx, directory)
	if err != nil {
		return "not paired: HEAD's benchmark does not build", nil
	}
	binaries := [sides]string{headSide: head, treeSide: tree}
	var runs [sides][]sliceRecord
	for range pairedRounds {
		for _, side := range abba {
			record, missing, err := l.runSlice(ctx, directory, binaries[side])
			if err != nil {
				return "", err
			}
			if missing != "" {
				if side == treeSide && strings.HasPrefix(missing, sliceFailed) && len(runs[headSide]) != 0 {
					return "", treeFailure(missing)
				}
				return "not paired: " + sideNames[side] + " " + missing, nil
			}
			if record.Identity != identity {
				return fmt.Sprintf("not paired: %s measures %s under another identity", sideNames[side], record.Model), nil
			}
			runs[side] = append(runs[side], record)
		}
	}
	var measured, worse []string
	for _, timed := range predicted {
		change, noise := pairedChange(runs, timed.name)
		cell := fmt.Sprintf("%s %+.1f%% (noise %.1f%%)", timed.name, change*percent, noise*percent)
		measured = append(measured, cell)
		if (change > 0) == timed.lowerIsBetter && math.Abs(change) > noise {
			worse = append(worse, cell)
		}
	}
	if len(worse) != 0 {
		return "", fmt.Errorf("the predicted effect names %s, which paired runs of HEAD and the tree measure worse beyond the noise of same-binary runs", strings.Join(worse, ", "))
	}
	return fmt.Sprintf("paired over %d rounds: %s", pairedRounds, strings.Join(measured, ", ")), nil
}

// pairedChange is the tree's relative change in the metric against HEAD,
// between the means of each side's runs, and the noise: the wider side's
// range over all its runs, relative to its mean. A slowdown that lands on
// one side's runs of one round widens that side's range with it.
func pairedChange(runs [sides][]sliceRecord, name metric.Name) (float64, float64) {
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

// sliceBaseline is the newest first-parent landing whose slice has the
// identity, and its commit; none is an empty record.
func (l *landing) sliceBaseline(ctx context.Context, identity string) (sliceRecord, string, error) {
	log, err := gitauthority.Query(ctx, l.Repo, "log", "--first-parent", "--grep=^"+sliceTrailer+": ",
		"--format=%h %(trailers:key="+sliceTrailer+",valueonly,separator=%x20)")
	if err != nil {
		return sliceRecord{}, "", err
	}
	for line := range strings.Lines(string(log)) {
		commit, value, _ := strings.Cut(strings.TrimSpace(line), " ")
		var record sliceRecord
		if json.Unmarshal([]byte(value), &record) == nil && record.Identity == identity {
			return record, commit, nil
		}
	}
	return sliceRecord{}, "", nil
}

// describeSlice says what the slice measured: the exact cells against the
// baseline when one exists, then the timed cells as measured.
func describeSlice(record, baseline sliceRecord, commit string) string {
	peak := record.Metrics[metric.PeakMemory] / bytesPerMiB
	var outputs string
	switch {
	case record.Outputs == variedOutputs:
		outputs = "outputs vary between runs"
	case commit == "":
		outputs = "outputs recorded"
	case baseline.Outputs == variedOutputs:
		outputs = "outputs not compared: the baseline's varied"
	case record.Outputs == baseline.Outputs:
		outputs = "outputs identical"
	default:
		outputs = "outputs differ"
	}
	exact := fmt.Sprintf("no earlier slice of this identity: %s; %s %.1f MiB", outputs, metric.PeakMemory, peak)
	if commit != "" {
		exact = fmt.Sprintf("against %s: %s; %s %.1f MiB (%+.1f)", commit, outputs, metric.PeakMemory, peak,
			peak-baseline.Metrics[metric.PeakMemory]/bytesPerMiB)
	}
	return fmt.Sprintf("%s, %s; unpaired %s %.0f ms, %s %.0f tok/s, %s %.0f ms", record.Model, exact,
		metric.FirstOutput, record.Metrics[metric.FirstOutput], metric.Throughput, record.Metrics[metric.Throughput],
		metric.Load, record.Metrics[metric.Load])
}
