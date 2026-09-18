//go:build windows

package mediacapability

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func TestFrozenKreaGeneration(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_KREA_BASELINE") != "1" {
		t.Skip("set OVERGO_KREA_BASELINE=1 for the frozen Krea image")
	}
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := overgodb.OpenReadOnly(cmp.Or(os.Getenv("OVERGO_SMOKE_REFERENCE_STORE"), roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := reference.Close(); err != nil {
			t.Error(err)
		}
	})
	inputID, err := artifact.ParseID("file:sha256:1d840594046f2981e9fb7c335808a4f3399101072ab0a8ab2768492642f9b6e2")
	if err != nil {
		t.Fatal(err)
	}
	input, present, err := artifact.ReadContent(t.Context(), reference, inputID)
	if err != nil || !present {
		t.Fatalf("required frozen request: present=%v error=%v", present, err)
	}
	if err := input.Validate(); err != nil {
		t.Fatal(err)
	}
	var request latentimage.Request
	if err := strictjson.DecodeBytes(input.Data, &request); err != nil {
		t.Fatal(err)
	}
	// Accepted reacquisition and review: docs/image_video_latent_images.json.
	const outputHash = "4508f19a532591c01d1c9c5b1e783b21f593484e7debf93e104e3f0df41ccb44"
	want, err := os.ReadFile(filepath.Join(root, "docs/media_samples", outputHash+".png"))
	if err != nil {
		t.Fatal(err)
	}
	directory := testutil.ModelArtifactDir(t, "Krea-2-Turbo")
	source, err := resolveImageSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	modelID := source.Inventory.Manifest.ID
	if modelID.String() != "model:sha256:e384ce5a7ff89dab915e5e636393f93e2a8f2cde2f0923e2e5df4363dda481c5" {
		t.Fatal("frozen model changed", modelID)
	}
	definition, contents, err := source.Define(modelID)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := latentimage.ResolveProfile(directory)
	if err != nil {
		t.Fatal(err)
	}
	if definition.ID.String() != "recipe:sha256:0b1bcc2fb9cf5f42cd2dd9c91a49dcce1a67c7b5a9afa0434aeddbe721d595d9" {
		t.Fatal("frozen component recipe or profile changed", definition.ID)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	// Current source declares prepare/text, integrate/transformer, decode/VAE.
	if len(source.Manifests) != len(program.Stages()) {
		t.Fatal("current component recipe was not selected")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	batch, err := source.Inventory.Batch(t.Name() + "/inventory")
	if err != nil {
		t.Fatal(err)
	}
	batch.Manifests = append(batch.Manifests, source.Manifests...)
	batch.Contents = append(batch.Contents, contents...)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	executor, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	for index, stage := range program.Stages() {
		if executor.ModuleModel(stage.Module.ID, modelID) != source.Manifests[index].ID {
			t.Fatalf("stage %s lost its component binding", stage.Node.ID)
		}
	}
	loaded := time.Now()
	generator, err := latentimage.LoadGenerator(t.Context(), directory, profile, request)
	if err != nil {
		t.Fatal(err)
	}
	loadWall := time.Since(loaded)
	t.Cleanup(func() {
		if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	if err := latentimage.RegisterRuntime(executor, modelID, generator); err != nil {
		t.Fatal(err)
	}
	operation, err := workflowruntime.ExecutionID(definition.ID, t.Name())
	if err != nil {
		t.Fatal(err)
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	started := time.Now()
	result, err := executor.ExecuteProgram(t.Context(), t.Name(), operation, nil, program, map[recipe.PortName]workflowruntime.Value{definition.Inputs[0].Name: workflowruntime.ArtifactValue(definition.Inputs[0].Data, request, input)})
	wall := time.Since(started)
	runtime.ReadMemStats(&after)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run.Outcome != runrecord.OutcomeSucceeded || result.Run.Recipe != definition.ID || len(result.Run.Inputs) != 1 || result.Run.Inputs[0] != inputID || len(result.Run.Outputs) != 1 || result.Run.Outputs[0].String() != "output:sha256:"+outputHash {
		t.Fatalf("frozen lineage changed: %+v", result.Run)
	}
	value, ok := result.Outputs[definition.Outputs[0].Name].Single()
	if !ok {
		t.Fatal("missing scalar image")
	}
	image, ok := value.Value.(latentimage.EncodedImage)
	if !ok || image.Width != request.Width || image.Height != request.Height || !bytes.Equal(image.Data, want) {
		t.Fatal("frozen pixels or geometry changed")
	}
	// An empty test store and new runtime require actual stage execution.
	// The shared PNG encoder rejects non-finite floats before quantization.
	if len(result.NodeWalls) != len(program.Stages()) {
		t.Fatal("missing current execution phases")
	}
	if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
		t.Fatal(err)
	}
	if err := generator.Reset(t.Context(), request); err == nil {
		t.Fatal("closed generator admitted another request")
	}
	row, err := json.Marshal(map[string]any{"case": "Krea-2-Turbo/image-gen/1", "model": modelID, "recipe": definition.ID, "components": source.Manifests, "input": inputID, "output": result.Run.Outputs[0], "run": result.Run.ID, "request": request, "load_ns": loadWall.Nanoseconds(), "request_ns": wall.Nanoseconds(), "allocated_bytes": after.TotalAlloc - before.TotalAlloc, "phases": result.NodeWalls, "scope": "Current component recipe, immutable request and accepted image. Shared GPU correctness. Request includes temporary-store publication, excludes load/close. Host allocations are process deltas. Native distribution and resource recovery remain separate owner acceptance; no quality or speedup claim."})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("CURRENT_MEDIA %s", row)
	t.Log("resource: task=Krea state=not_busy scope=closed-generator")
}
