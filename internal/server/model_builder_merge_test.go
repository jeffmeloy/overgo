package server

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/export"
	"overgo/internal/gguf"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/quant"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

// ggufTensorValues decodes one tensor of a GGUF file.
func ggufTensorValues(t *testing.T, path, name string) []float32 {
	t.Helper()
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	index := slices.IndexFunc(file.Tensors, func(info gguf.TensorInfo) bool { return info.Name == name })
	if index < 0 {
		t.Fatalf("%s holds no tensor %q", path, name)
	}
	info := file.Tensors[index]
	raw := make([]byte, info.Size)
	if err := file.ReadTensorData(info, raw); err != nil {
		t.Fatal(err)
	}
	elements, err := info.ElementCount()
	if err != nil {
		t.Fatal(err)
	}
	values, err := quant.Dequantize(info.Type, raw, elements)
	if err != nil {
		t.Fatal(err)
	}
	return values
}

// mergeFixture catalogues the hermetic model and a fine-tuned model sharing
// its tensor inventory with every weight one higher.
func mergeFixture(t *testing.T, store *overgodb.Store) (base, tuned string) {
	t.Helper()
	base = testutil.HermeticLlamaGGUF(t, 64)
	tuned = filepath.Join(t.TempDir(), "tuned.gguf")
	source, err := gguf.Open(base)
	if err != nil {
		t.Fatal(err)
	}
	destination, err := os.Create(tuned)
	if err != nil {
		t.Fatal(err)
	}
	if err := source.RewriteTo(destination, gguf.DTypeF32, func(_ gguf.TensorInfo, values []float32) error {
		for index := range values {
			values[index]++
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := errors.Join(source.Close(), destination.Close()); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{base, tuned} {
		if err := servingtest.PublishActiveGGUFWithPolicy(t.Context(), store, path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference); err != nil {
			t.Fatal(err)
		}
	}
	return base, tuned
}

// TestModelBuilderMergeRun: Model Builder offers a merge onto each catalogued
// GGUF model with the others as the fine-tuned choice, beside the adapter
// fold with its refusal; a merge writes base + scale * (fine-tuned - base) as
// an F32 GGUF under the output and records a run with both models in and the
// merged file out at its location; an unknown fine-tuned model, an existing
// output and the adapter fold are refused.
func TestModelBuilderMergeRun(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	base, tuned := mergeFixture(t, store)
	root := t.TempDir()
	workspace := NewGGUFWorkspace(store, root, 64, GGUFWriters{Types: export.GGUFTypes(), Export: export.GGUF, Merge: export.MergeGGUF}, "")
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowModelBuild)
	if err != nil || len(capabilities) != 3 {
		t.Fatalf("model builder capabilities = %+v, %v", capabilities, err)
	}
	merge := capabilities[slices.IndexFunc(capabilities, func(capability WorkflowCapability) bool { return capability.Name == "hermetic.gguf" })]
	fold := capabilities[len(capabilities)-1]
	if merge.Task != recipe.TaskMerge || !slices.Equal(merge.Controls[0].Choices, []string{"tuned.gguf"}) || fold.Refusal == "" {
		t.Fatalf("merge = %+v fold = %+v", merge, fold)
	}
	execute := func(capability WorkflowCapability, input string) (runrecord.Run, error) {
		t.Helper()
		completion, err := workspace.ExecuteWorkflow(ctx, workflowModelBuild, capability, json.RawMessage(input), &recordingReporter{})
		if err != nil {
			return runrecord.Run{}, err
		}
		return runrecord.RequireRun(ctx, store, completion.Run)
	}
	run, err := execute(merge, `{"fine_tuned":"tuned.gguf","scale":0.5,"output":"merged"}`)
	if err != nil || len(run.Inputs) != 2 || run.Inputs[0] != merge.Model || len(run.Outputs) != 1 {
		t.Fatalf("merge run = %+v, %v", run, err)
	}
	merged := filepath.Join(root, "merged", "hermetic.merged.gguf")
	baseValues, tunedValues, mergedValues := ggufTensorValues(t, base, "output.weight"), ggufTensorValues(t, tuned, "output.weight"), ggufTensorValues(t, merged, "output.weight")
	for index := range baseValues {
		want := baseValues[index] + 0.5*(tunedValues[index]-baseValues[index])
		if math.Abs(float64(mergedValues[index]-want)) > 1e-6 {
			t.Fatalf("merged output.weight[%d] = %v, want %v", index, mergedValues[index], want)
		}
	}
	locations, err := store.Locations(ctx, run.Outputs[0])
	if err != nil || !slices.ContainsFunc(locations, func(location artifact.Location) bool { return location.Value == merged }) {
		t.Fatalf("merged locations = %+v, %v", locations, err)
	}
	if _, err := execute(merge, `{"fine_tuned":"absent.gguf","scale":1,"output":"other"}`); err == nil || !strings.Contains(err.Error(), `0 catalogued models are named "absent.gguf"`) {
		t.Fatalf("an unknown fine-tuned model: %v", err)
	}
	if _, err := execute(merge, `{"fine_tuned":"tuned.gguf","scale":1,"output":"merged"}`); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("an existing output: %v", err)
	}
	if _, err := execute(fold, `{}`); err == nil || !strings.Contains(err.Error(), "no implementation yet") {
		t.Fatalf("the adapter fold: %v", err)
	}
}
