package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/strictjson"
)

// sliceDocument is one scorecard slice: the task it measures on that task's
// smallest local model, how many runs it takes, and its fixed input. The
// slice is named by the document's file name.
type sliceDocument struct {
	Task    recipe.Task `json:"task"`
	Runs    int         `json:"runs"`
	Warmup  int         `json:"warmup,omitzero"`
	Prompts []string    `json:"prompts,omitempty"`
}

// runSlices measures each document in turn and writes two record lines for
// each: what it selected once that is known, then its outcome. A slice that
// cannot run is a record too, so one slice never keeps the next from
// measuring; only a crash ends the process, and the selected line already
// written says which slice it was measuring.
func runSlices(ctx context.Context, options options, output io.Writer) error {
	encoder := json.NewEncoder(output)
	for _, document := range options.Documents {
		name := strings.TrimSuffix(filepath.Base(document), filepath.Ext(document))
		record := measureSlice(ctx, options, name, document, func(selected metric.Record) error {
			selected.Slice = name
			return encoder.Encode(selected)
		})
		record.Slice = name
		if err := encoder.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

// measureSlice runs one document and returns its outcome record.
func measureSlice(ctx context.Context, options options, name, path string, selected func(metric.Record) error) metric.Record {
	data, err := os.ReadFile(path)
	if err != nil {
		return failedSlice(metric.Record{}, err)
	}
	var document sliceDocument
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return failedSlice(metric.Record{}, fmt.Errorf("slice %s: %w", name, err))
	}
	if document.Runs < minBenchmarkRuns || document.Runs > maxBenchmarkRuns || document.Warmup < minBenchmarkWarmup || document.Warmup > maxBenchmarkWarmup {
		return failedSlice(metric.Record{}, fmt.Errorf("slice %s: runs must be in [%d,%d] and warmup in [%d,%d]", name, minBenchmarkRuns, maxBenchmarkRuns, minBenchmarkWarmup, maxBenchmarkWarmup))
	}
	switch document.Task {
	case recipe.TaskInference:
		return measureInferenceSlice(ctx, options, document, selected)
	default:
		return failedSlice(metric.Record{}, fmt.Errorf("slice %s: no slice runner measures task %q", name, document.Task))
	}
}

// measureInferenceSlice measures the smallest local GGUF on the document's
// prompts, with the benchmark's defaults and the document's runs.
func measureInferenceSlice(ctx context.Context, options options, document sliceDocument, selected func(metric.Record) error) metric.Record {
	if err := validPrompts(document.Prompts); err != nil {
		return failedSlice(metric.Record{}, err)
	}
	model, repository, err := smallestModel(ctx, options.Repository)
	if err != nil {
		return failedSlice(metric.Record{}, err)
	}
	options.Model, options.Repository = model, repository
	options.Runs, options.Warmup = document.Runs, document.Warmup
	var chosen metric.Record
	var announced error
	result, _, _, err := measureArms(options, document.Prompts, func(loaded benchmarkResult) {
		chosen = metric.Record{
			Identity: sliceIdentity(loaded),
			Model:    loaded.ModelName,
		}
		announced = selected(chosen)
	})
	if err = errors.Join(err, announced); err != nil {
		return failedSlice(chosen, err)
	}
	return inferenceRecord(chosen, result)
}

// inferenceRecord maps a benchmark result onto the slice's record: one
// output digest when every run agrees, and the scorecard's timed cells.
func inferenceRecord(chosen metric.Record, result benchmarkResult) metric.Record {
	outputs := metric.VariedOutputs
	if len(result.Runs) != 0 && !slices.ContainsFunc(result.Runs, func(run runMetrics) bool {
		return !slices.Equal(run.OutputDigests, result.Runs[0].OutputDigests)
	}) {
		outputs = strings.Join(result.Runs[0].OutputDigests, ",")
	}
	chosen.Outputs = outputs
	chosen.Metrics = map[metric.Name]float64{
		metric.FirstOutput: result.Summary.TTFTMillisecondsP50,
		metric.Throughput:  result.Summary.DecodeTokensPerSecondP50,
		metric.Load:        result.LoadMilliseconds,
		metric.PeakMemory:  float64(result.DevicePeakBytes),
		metric.Failures:    0,
	}
	return chosen
}

// failedSlice is the record of a slice that did not measure: unavailable
// when the device is claimed or out of memory, which says nothing of the
// code, and failed otherwise, with the cause on one line and a failure-rate
// of one. It keeps what the slice selected, if it got that far.
func failedSlice(chosen metric.Record, err error) metric.Record {
	cause := strings.Join(strings.Fields(err.Error()), " ")
	if errors.Is(err, processcontrol.ErrResourceBusy) || errors.Is(err, processcontrol.ErrDeviceMemory) {
		chosen.Unmeasured = "unavailable: " + cause
		return chosen
	}
	chosen.Failure = cause
	chosen.Metrics = map[metric.Name]float64{metric.Failures: 1}
	return chosen
}

// sliceIdentity is what a slice's measurement is comparable under: the
// model, recipe, workload, options and environment, digested so a trailer
// carries it on one line.
func sliceIdentity(result benchmarkResult) string {
	sum := sha256.Sum256([]byte(strings.Join([]string{
		result.ModelID.String(), result.RecipeID.String(), result.WorkloadDigest, result.OptionsDigest, result.EnvironmentID.String(),
	}, "\n")))
	return hex.EncodeToString(sum[:])
}
