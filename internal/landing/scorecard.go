package landing

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
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

// slice is what the scorecard slice depends on: the benchmark binary's
// closure, which embeds the built kernels, the module versions and the
// workload.
var slice = lane{name: "scorecard", packages: []string{"./cmd/benchmark"},
	prefixes: []string{"go.mod", "go.sum", sliceWorkload}}

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
// options and environment. The timed cells are reported as measured and
// unpaired: one run on each side cannot tell a change from noise. The slice
// never refuses a landing: a run that does not measure says why, as
// unavailable when the device is claimed or out of memory and as failed
// otherwise, since only a run of HEAD beside it could blame the change.
func (l *landing) scorecard(ctx context.Context) error {
	reached, err := l.reaches(ctx, slice)
	if err != nil {
		return err
	}
	if !reached {
		l.details["scorecard"] = "not reached"
		return nil
	}
	record, missing, err := l.measureSlice(ctx)
	if err != nil {
		return err
	}
	if missing != "" {
		l.details["scorecard"] = missing
		return nil
	}
	baseline, commit, err := l.sliceBaseline(ctx, record.Identity)
	if err != nil {
		return err
	}
	l.details["scorecard"] = describeSlice(record, baseline, commit)
	encoded, err := json.Marshal(record)
	if err != nil {
		return err
	}
	l.trailer = sliceTrailer + ": " + string(encoded)
	return nil
}

// measureSlice builds the benchmark from the tree and runs it once, so its
// exit status is the benchmark's own. A run that does not measure returns
// why, in one line.
func (l *landing) measureSlice(ctx context.Context) (sliceRecord, string, error) {
	directory, err := os.MkdirTemp("", "overgo-scorecard-*")
	if err != nil {
		return sliceRecord{}, "", err
	}
	defer os.RemoveAll(directory)
	binary := filepath.Join(directory, "benchmark")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if _, err := l.run(ctx, nil, "go", "build", "-o", binary, "./cmd/benchmark"); err != nil {
		return sliceRecord{}, "failed: the benchmark does not build", nil
	}
	errorFile := filepath.Join(directory, "error.txt")
	var output, diagnostics bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: binary, Args: []string{"-smallest", "-prompt-suite", sliceWorkload}, Dir: l.Repo,
		Env:    append(os.Environ(), clioptions.ErrorFileEnvironment+"="+errorFile),
		Stdout: &output, Stderr: &diagnostics,
	})
	if err != nil {
		return sliceRecord{}, "", err
	}
	switch receipt.ExitCode {
	case 0:
	case processcontrol.ResourceBusyExitCode, processcontrol.DeviceMemoryExitCode:
		return sliceRecord{}, "unavailable: " + failureCause(errorFile, diagnostics.String()), nil
	default:
		return sliceRecord{}, fmt.Sprintf("failed: exit %d: %s", receipt.ExitCode, failureCause(errorFile, diagnostics.String())), nil
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
		return sliceRecord{}, "failed: the benchmark printed no measured runs", nil
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
