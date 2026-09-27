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
	workspace := NewExportWorkspace(store, root, 64, GGUFExporter{Types: export.GGUFTypes(), Write: export.GGUF})
	capabilities, err := workspace.WorkflowCapabilities(ctx, WorkflowExport)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("export capabilities = %+v, %v", capabilities, err)
	}
	capability := capabilities[0]
	choices := capability.Controls[0].Choices
	if capability.Task != recipe.TaskExport || capability.Name != "hermetic.gguf" || !slices.Contains(choices, "f16") || !slices.Contains(choices, "q8_0") || slices.Contains(choices, "iq2_xxs") {
		t.Fatalf("export capability = %+v", capability)
	}
	run := func(input string) error {
		_, err := workspace.ExecuteWorkflow(ctx, WorkflowExport, recipe.TaskExport, capability.Recipe, json.RawMessage(input), &recordingReporter{})
		return err
	}

	reporter := &recordingReporter{}
	completion, err := workspace.ExecuteWorkflow(ctx, WorkflowExport, recipe.TaskExport, capability.Recipe, json.RawMessage(`{"type":"f16","output":"shared"}`), reporter)
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
	locations, err := store.Locations(ctx, completion.Outputs[0])
	if err != nil || !slices.ContainsFunc(locations, func(location artifact.Location) bool {
		return location.Kind == artifact.LocationFile && location.Value == file
	}) {
		t.Fatalf("exported locations = %+v, %v", locations, err)
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
