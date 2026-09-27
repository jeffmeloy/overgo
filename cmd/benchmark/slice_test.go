package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

// TestSliceRecordKeepsTheLandedIdentity pins the identity a slice record
// carries to the digest landings have compared by, and the mapping of a
// benchmark result onto the scorecard's cells.
func TestSliceRecordKeepsTheLandedIdentity(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "environment")
	sum := sha256.Sum256([]byte(strings.Join([]string{model.String(), recipeID.String(), "workload", "options", environment.String()}, "\n")))
	loaded := benchmarkResult{ModelID: model, RecipeID: recipeID, WorkloadDigest: "workload", OptionsDigest: "options", EnvironmentID: environment}
	if got, want := sliceIdentity(loaded), hex.EncodeToString(sum[:]); got != want {
		t.Fatalf("identity = %s, want %s", got, want)
	}
	result := benchmarkResult{
		LoadMilliseconds: 4000, DevicePeakBytes: 1 << 20,
		Runs:    []runMetrics{{OutputDigests: []string{"a"}}, {OutputDigests: []string{"a"}}},
		Summary: summaryMetrics{TTFTMillisecondsP50: 150, DecodeTokensPerSecondP50: 450},
	}
	record := inferenceRecord(metric.Record{Identity: "identity", Model: "small"}, result)
	want := map[metric.Name]float64{metric.FirstOutput: 150, metric.Throughput: 450, metric.Load: 4000, metric.PeakMemory: 1 << 20, metric.Failures: 0}
	if record.Outputs != "a" || record.Identity != "identity" || fmt.Sprint(record.Metrics) != fmt.Sprint(want) {
		t.Fatalf("record = %+v, want outputs a and metrics %v", record, want)
	}
	result.Runs[1].OutputDigests = []string{"b"}
	if record := inferenceRecord(metric.Record{}, result); record.Outputs != metric.VariedOutputs {
		t.Fatalf("disagreeing runs gave outputs %q", record.Outputs)
	}
}

// TestSliceClassifiesEachOutcome holds a slice that cannot run to a record,
// never a process error: a claimed or exhausted device is unavailable and
// says nothing of the code; any other error is a failure on one line with
// a failure-rate of one, keeping what the slice had selected.
func TestSliceClassifiesEachOutcome(t *testing.T) {
	t.Parallel()
	selected := metric.Record{Identity: "identity", Model: "small"}
	for _, err := range []error{processcontrol.ErrResourceBusy, fmt.Errorf("load: %w", processcontrol.ErrDeviceMemory)} {
		if record := failedSlice(selected, err); !strings.HasPrefix(record.Unmeasured, "unavailable: ") || record.Failure != "" || record.Metrics != nil {
			t.Fatalf("%v gave %+v, want unavailable", err, record)
		}
	}
	record := failedSlice(selected, errors.New("decode:\n  panic in\tlayer 3"))
	if record.Failure != "decode: panic in layer 3" || record.Metrics[metric.Failures] != 1 || record.Identity != "identity" {
		t.Fatalf("failure record = %+v", record)
	}
	directory := t.TempDir()
	for name, document := range map[string]string{
		"forecast": `{"task":"forecast","runs":2}`,
		"broken":   `{"task":"inference","runs":0,"prompts":["a"]}`,
		"unknown":  `{"task":"inference","runs":1,"prompts":["a"],"extra":1}`,
	} {
		path := filepath.Join(directory, name+".json")
		if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
			t.Fatal(err)
		}
		record := measureSlice(t.Context(), options{}, name, path, func(metric.Record) error { return nil })
		if record.Failure == "" || record.Metrics[metric.Failures] != 1 {
			t.Fatalf("%s gave %+v, want a failed record", name, record)
		}
	}
}

// TestSliceDocumentsDecode holds every committed slice document to the
// strict schema, and the inference document to the prompt the landed
// inference slice measured, so its identity and baseline carry over.
func TestSliceDocumentsDecode(t *testing.T) {
	t.Parallel()
	documents, err := filepath.Glob(filepath.Join("..", "..", "fixtures", "scorecard_slice", "*.json"))
	if err != nil || len(documents) == 0 {
		t.Fatalf("no slice documents: %v", err)
	}
	for _, path := range documents {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var document sliceDocument
		if err := strictjson.DecodeBytes(data, &document); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if document.Runs < minBenchmarkRuns || document.Task == "" || (document.Task == recipe.TaskInference) != (len(document.Prompts) != 0) {
			t.Fatalf("%s = %+v, want a task, runs, and prompts only for inference", path, document)
		}
		// The workload the landed inference slice measured (ee13bdab2).
		const landedWorkload = "ef348b0e5d6a1d75970d369242a0fe37ccd13abdb67bcb5e2d2ae4d53f6b2da9"
		if document.Task == recipe.TaskInference && benchmarkWorkloadDigest(document.Prompts) != landedWorkload {
			t.Fatalf("%s: workload digest %s, want the landed %s", path, benchmarkWorkloadDigest(document.Prompts), landedWorkload)
		}
	}
}
