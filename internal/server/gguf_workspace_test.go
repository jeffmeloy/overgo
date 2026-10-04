package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/export"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingtest"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testutil"
)

// TestExportQuantizedGGUF: every catalogued GGUF model is an export
// capability whose types are those written without an importance matrix;
// an export writes the quantized GGUF and an Ollama Modelfile naming it in a
// new directory under the root, and records a run with the model in and the
// written file out at its location; an existing output, an importance type
// and an output outside the root are refused and leave nothing behind.
func TestExportQuantizedGGUF(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	source := testutil.HermeticLlamaGGUF(t, 64)
	if err := servingtest.PublishActiveGGUFWithPolicy(ctx, store, source, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	workspace := NewGGUFWorkspace(store, root, 64, GGUFWriters{Types: export.GGUFTypes(), Export: export.GGUF, Merge: export.MergeGGUF}, "")
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowExport)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("export capabilities = %+v, %v", capabilities, err)
	}
	capability := capabilities[0]
	choices := capability.Controls[0].Choices
	if capability.Task != recipe.TaskExport || capability.Name != "hermetic.gguf" || !slices.Contains(choices, "f16") || !slices.Contains(choices, "q8_0") || slices.Contains(choices, "iq2_xxs") {
		t.Fatalf("export capability = %+v", capability)
	}
	run := func(input string) error {
		_, err := workspace.ExecuteWorkflow(ctx, workflowExport, capability, json.RawMessage(input), &recordingReporter{})
		return err
	}

	reporter := &recordingReporter{}
	completion, err := workspace.ExecuteWorkflow(ctx, workflowExport, capability, json.RawMessage(`{"type":"f16","output":"shared"}`), reporter)
	if err != nil || !reporter.published || len(completion.Outputs) != 1 {
		t.Fatalf("export = %+v published=%v err=%v", completion, reporter.published, err)
	}
	file := filepath.Join(root, "shared", "hermetic.f16.gguf")
	written, err := gguf.Open(file)
	if err != nil {
		t.Fatal(err)
	}
	converted := slices.ContainsFunc(written.Tensors, func(info gguf.TensorInfo) bool { return info.Name == "output.weight" && info.Type == dtype.F16 })
	_ = written.Close()
	if !converted {
		t.Fatal("the exported GGUF does not hold its weights as f16")
	}
	if modelfile, err := os.ReadFile(filepath.Join(root, "shared", "Modelfile")); err != nil || string(modelfile) != "FROM ./hermetic.f16.gguf\n" {
		t.Fatalf("Modelfile = %q, %v", modelfile, err)
	}
	record, err := runrecord.RequireRun(ctx, store, completion.Run)
	if err != nil || !slices.Equal(record.Inputs, []artifact.ID{capability.Model}) || !slices.Equal(record.Outputs, completion.Outputs) {
		t.Fatalf("export run = %+v err=%v", record, err)
	}
	inventory, err := modelartifact.FromGGUFPath(file)
	if err != nil || completion.Outputs[0] != inventory.Manifest.ID {
		t.Fatalf("the run's output is %s, want the identity every intake gives the written model, %s: %v", completion.Outputs[0], inventory.Manifest.ID, err)
	}
	if want, err := artifact.CanonicalLocalLocation(completion.Outputs[0], artifact.LocationDirectory, filepath.Dir(file)); err != nil {
		t.Fatal(err)
	} else if locations, err := store.Locations(ctx, completion.Outputs[0]); err != nil || !slices.Contains(locations, want) {
		t.Fatalf("exported locations = %+v, want %+v: %v", locations, want, err)
	}

	for input, refusal := range map[string]string{
		`{"type":"f16","output":"shared"}`:  "already exists",
		`{"type":"iq2_xxs","output":"imp"}`: "without an importance matrix",
		`{"type":"f16","output":"../out"}`:  "one path element",
	} {
		if err := run(input); err == nil || !strings.Contains(err.Error(), refusal) {
			t.Fatalf("export %s: err=%v, want %q", input, err, refusal)
		}
	}
	for _, left := range []string{"imp", "out"} {
		if _, err := os.Stat(filepath.Join(root, left)); !os.IsNotExist(err) {
			t.Fatalf("a refused export left %s: %v", left, err)
		}
	}
}

// ggufTestCommit stands in for the executable's clean source revision.
const ggufTestCommit = "0123456789abcdef0123456789abcdef01234567"

// TestGGUFRunsBindTheirSource: when the launcher names the executable's clean
// source commit, an export's run binds that commit, this host's published
// environment and the write's measured wall; without one the run records no
// binding.
func TestGGUFRunsBindTheirSource(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := servingtest.PublishActiveGGUFWithPolicy(ctx, store, testutil.HermeticLlamaGGUF(t, 64), recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference); err != nil {
		t.Fatal(err)
	}
	exportRun := func(commit, output string) runrecord.Run {
		t.Helper()
		workspace := NewGGUFWorkspace(store, t.TempDir(), 64, GGUFWriters{Types: export.GGUFTypes(), Export: export.GGUF, Merge: export.MergeGGUF}, commit)
		capabilities, err := workspace.WorkflowCapabilities(ctx, workflowExport)
		if err != nil || len(capabilities) != 1 {
			t.Fatalf("export capabilities = %+v, %v", capabilities, err)
		}
		completion, err := workspace.ExecuteWorkflow(ctx, workflowExport, capabilities[0],
			json.RawMessage(`{"type":"f16","output":"`+output+`"}`), &recordingReporter{})
		if err != nil {
			t.Fatal(err)
		}
		run, err := runrecord.RequireRun(ctx, store, completion.Run)
		if err != nil {
			t.Fatal(err)
		}
		return run
	}
	bound := exportRun(ggufTestCommit, "bound")
	environment, err := runrecord.RequireEnvironment(ctx, store, bound.Environment)
	if err != nil || bound.CodeCommit != ggufTestCommit || bound.MeasuredNS == 0 || environment.Device != "host" || environment.Backend != "go" {
		t.Fatalf("bound run = commit %q environment %s measured %d: %+v, %v", bound.CodeCommit, bound.Environment, bound.MeasuredNS, environment, err)
	}
	if unbound := exportRun("", "unbound"); unbound.CodeCommit != "" || unbound.Environment.Valid() || unbound.MeasuredNS != 0 {
		t.Fatalf("unbound run = %+v", unbound)
	}
}

// TestRepeatExportKeepsItsOutput holds an export of bytes already exported
// under another name to a second location of the same model: the run, which
// without a bound commit is the same record, is not a conflict, and neither
// finished export is removed for the other.
func TestRepeatExportKeepsItsOutput(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := servingtest.PublishActiveGGUFWithPolicy(ctx, store, testutil.HermeticLlamaGGUF(t, 64), recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	workspace := NewGGUFWorkspace(store, root, 64, GGUFWriters{Types: export.GGUFTypes(), Export: export.GGUF, Merge: export.MergeGGUF}, "")
	capabilities, err := workspace.WorkflowCapabilities(ctx, workflowExport)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("export capabilities = %+v, %v", capabilities, err)
	}
	var outputs []artifact.ID
	for _, name := range []string{"first", "second"} {
		completion, err := workspace.ExecuteWorkflow(ctx, workflowExport, capabilities[0],
			json.RawMessage(`{"type":"f16","output":"`+name+`"}`), &recordingReporter{})
		if err != nil {
			t.Fatalf("export %s: %v", name, err)
		}
		outputs = append(outputs, completion.Outputs...)
		if _, err := os.Stat(filepath.Join(root, name, "hermetic.f16.gguf")); err != nil {
			t.Fatalf("export %s left no file: %v", name, err)
		}
	}
	if len(outputs) != 2 || outputs[0] != outputs[1] {
		t.Fatalf("the same bytes were identified as %v", outputs)
	}
	locations, err := store.Locations(ctx, outputs[0])
	if err != nil || len(locations) != 2 {
		t.Fatalf("the model sits at %+v, want both exports: %v", locations, err)
	}
}

// TestModelBuilderListsNothingWithoutAMerge holds Model Builder's tab to what
// can run: one catalogued model has no other to merge, and the adapter fold's
// refusal beside merges is not a capability that enables the tab alone.
func TestModelBuilderListsNothingWithoutAMerge(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := servingtest.PublishActiveGGUFWithPolicy(ctx, store, testutil.HermeticLlamaGGUF(t, 64), recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference); err != nil {
		t.Fatal(err)
	}
	workspace := NewGGUFWorkspace(store, t.TempDir(), 64, GGUFWriters{Types: export.GGUFTypes(), Export: export.GGUF, Merge: export.MergeGGUF}, "")
	if capabilities, err := workspace.WorkflowCapabilities(ctx, workflowModelBuild); err != nil || len(capabilities) != 0 {
		t.Fatalf("one model lists %+v for Model Builder, %v", capabilities, err)
	}
}
