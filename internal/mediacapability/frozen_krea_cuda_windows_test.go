//go:build modeltest && windows

package mediacapability

import (
	"cmp"
	"context"
	"encoding/json"
	"math"
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
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func TestFrozenKreaGeneration(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_KREA_BASELINE") != "1" {
		t.Skip(testskip.Inapplicable + ": set OVERGO_KREA_BASELINE=1 for the frozen Krea image")
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
	// The accepted sample, with its run and review: docs/verification/image-video-coverage.json.
	// docs/image_video_latent_images.json keeps the acquisition of the picture it replaced.
	const outputHash = "4844851dc3b37f5ec51704cc74f4863f53077ff903efde231daa482d834fc327"
	want, err := os.ReadFile(filepath.Join(root, "assets", outputHash+".png"))
	if err != nil {
		t.Fatal(err)
	}
	directory := testutil.ModelArtifactDir(t, "Krea-2-Turbo")
	source, err := resolveImageSource(t.Context(), directory)
	if err != nil {
		t.Fatal(err)
	}
	modelID := source.Inventory.Manifest.ID
	if modelID.String() != "model:sha256:4ad720595938a39340a70f4298398a8201b5ea6e214883c9e166b4a0d3af8401" {
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
	if definition.ID.String() != "recipe:sha256:b935833ba39704b1086369c69bc453b453a0183411d36dcd144a08dd1b6969ee" {
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
	if result.Run.Outcome != runrecord.OutcomeSucceeded || result.Run.Recipe != definition.ID || len(result.Run.Inputs) != 1 || result.Run.Inputs[0] != inputID || len(result.Run.Outputs) != 1 {
		t.Fatalf("frozen lineage changed: %+v", result.Run)
	}
	value, ok := result.Outputs[definition.Outputs[0].Name].Single()
	if !ok {
		t.Fatal("missing scalar image")
	}
	// The picture is held to the frozen one, written before Krea's decoder grid was finished away (krea-degrid): each
	// value within the finish's 2-level limit and a rounding of the frozen picture's, its 1-pixel texture within 1% of
	// it, and its grid gone in flat areas, which the decoder's unrounded output lets the finish take.
	image, ok := value.Value.(latentimage.EncodedImage)
	if !ok || image.Width != request.Width || image.Height != request.Height {
		t.Fatal("frozen geometry changed")
	}
	frozenPixels, servedPixels := testutil.PNGRGB(t, want), testutil.PNGRGB(t, image.Data)
	if len(frozenPixels) != len(servedPixels) {
		t.Fatal("frozen pixels changed")
	}
	for index := range frozenPixels {
		if moved := int(servedPixels[index]) - int(frozenPixels[index]); moved > 3 || moved < -3 {
			t.Fatalf("a frozen value moved %d levels; want 3 at most, the grid finish's limit and a rounding", moved)
		}
	}
	frozenPlane, servedPlane := testutil.PNGLuminance(t, want), testutil.PNGLuminance(t, image.Data)
	frozenColumns, frozenRows := testutil.PixelGrid(frozenPlane, true)
	servedColumns, servedRows := testutil.PixelGrid(servedPlane, true)
	t.Logf("Krea's frozen picture, flat-block grid: frozen %.3f and %.3f levels, served %.3f and %.3f", frozenColumns, frozenRows, servedColumns, servedRows)
	// Measured with the finish on the decoder's unrounded output: 0.196 and 0.168 levels frozen, -0.014 and -0.052 served.
	if math.Abs(servedColumns) > 0.06 || math.Abs(servedRows) > 0.06 {
		t.Errorf("the served picture keeps a grid of %.3f and %.3f levels in its flat blocks; want under 0.06", servedColumns, servedRows)
	}
	if loss := 1 - testutil.PixelSteps(servedPlane)/testutil.PixelSteps(frozenPlane); math.Abs(loss) > 0.01 {
		t.Errorf("the served picture's 1-pixel texture is %.2f%% from the frozen one's; want within 1%%", 100*loss)
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
