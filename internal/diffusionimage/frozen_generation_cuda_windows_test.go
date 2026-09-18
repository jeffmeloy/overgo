//go:build windows

package diffusionimage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// Protocol: image_video_protocol.json; accepted samples: image_video_latent_images.json,
// acquisition 7a5f4da7. These noisy smoke images establish no semantic quality.
func TestFrozenDiffusionGeneration(t *testing.T) {
	cudatest.Require(t)
	directory := artifactDir(t)
	weights, err := filepath.Glob(filepath.Join(directory, "*.safetensors"))
	if err != nil || len(weights) != 1 {
		t.Fatalf("required single-checkpoint inventory: files=%v error=%v", weights, err)
	}
	inventory, err := modelartifact.FromFiles(directory, []modelartifact.FileSpec{
		{Path: filepath.Join(directory, "config.json"), Name: "config", Role: artifact.ComponentConfig},
		{Path: weights[0], Name: "weights", Role: artifact.ComponentWeights},
	})
	if err != nil {
		t.Fatal(err)
	}
	if inventory.Manifest.ID.String() != "model:sha256:5641a228dd9f9991ede664f1988615c62839896e51016777d1d301766278b939" {
		t.Fatal("frozen model changed", inventory.Manifest.ID)
	}
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleDiffusionImagePrepare, inventory.Manifest.ID, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	if definition.ID.String() != "recipe:sha256:4f0b3b9dbad24278519b1a046195978ab0960c52c0b4b40a2d52e0bcec873e9d" {
		t.Fatal("frozen recipe changed", definition.ID)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
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
	batch, err := inventory.Batch(t.Name() + "/inventory")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		request       Request
		input, output string
	}{
		{Request{Seed: 5, Steps: 8, Height: 256, Width: 256}, "ab444aa99af1a85d3b6a2b64fb7459892048e244112cc4e3f88101d797c377ea", "939df8925ae33a2064243a201d85659fae13ab700799f4796df2241aedcbda78"},
		{Request{Seed: 105, Steps: 8, Height: 256, Width: 256}, "2b9ae1a1da5bce737b548e20c27223dab9a351ba6e45dcd81c813fc63fc119dd", "e5b51b334d49a1fb74d4169c3f832c874fc7221f820fae6a7f6d127566880aef"},
	}
	generator, err := LoadResidentGenerator(t.Context(), directory, cases[0].request)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	executor, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	if err := RegisterRuntime(executor, inventory.Manifest.ID, generator); err != nil {
		t.Fatal(err)
	}
	for index, tc := range cases {
		t.Run(fmt.Sprintf("case-%d", index+1), func(t *testing.T) {
			input, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.image-gen-input.v1"), tc.request)
			if err != nil {
				t.Fatal(err)
			}
			if input.Descriptor.ID.String() != "file:sha256:"+tc.input {
				t.Fatal("frozen request changed")
			}
			want, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs/media_samples", tc.output+".png"))
			if err != nil {
				t.Fatal(err)
			}
			operation, err := workflowruntime.ExecutionID(definition.ID, t.Name())
			if err != nil {
				t.Fatal(err)
			}
			before, err := generator.forward.Stats(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			var hostBefore, hostAfter runtime.MemStats
			runtime.ReadMemStats(&hostBefore)
			started := time.Now()
			result, err := executor.ExecuteProgram(t.Context(), t.Name(), operation, nil, program, map[recipe.PortName]workflowruntime.Value{definition.Inputs[0].Name: workflowruntime.ArtifactValue(definition.Inputs[0].Data, tc.request, input)})
			wall := time.Since(started)
			runtime.ReadMemStats(&hostAfter)
			if err != nil {
				t.Fatal(err)
			}
			after, err := generator.forward.Stats(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if result.Run.Outcome != runrecord.OutcomeSucceeded || result.Run.Recipe != definition.ID || len(result.Run.Inputs) != 1 || result.Run.Inputs[0] != input.Descriptor.ID || len(result.Run.Outputs) != 1 || result.Run.Outputs[0].String() != "output:sha256:"+tc.output {
				t.Fatalf("frozen lineage changed: %+v", result.Run)
			}
			value, ok := result.Outputs[definition.Outputs[0].Name].Single()
			if !ok {
				t.Fatal("missing scalar image")
			}
			image, ok := value.Value.(latentimage.EncodedImage)
			if !ok || image.Width != tc.request.Width || image.Height != tc.request.Height || !bytes.Equal(image.Data, want) {
				t.Fatal("frozen image pixels or geometry changed")
			}
			// EncodePlanarPNG rejects every non-finite float before quantization.
			// Require fresh device execution; a stored output alone cannot pass.
			if after.Execution.KernelLaunches <= before.Execution.KernelLaunches || after.Execution.DeviceToHostBytes <= before.Execution.DeviceToHostBytes {
				t.Fatal("request performed no GPU work")
			}
			measurement, err := json.Marshal(map[string]any{"case": t.Name(), "model": inventory.Manifest.ID, "recipe": definition.ID, "input": input.Descriptor.ID, "output": result.Run.Outputs[0], "run": result.Run.ID, "request_ns": wall.Nanoseconds(), "allocated_bytes": hostAfter.TotalAlloc - hostBefore.TotalAlloc, "before": before, "after": after, "phases": result.NodeWalls, "minimum": image.Minimum, "maximum": image.Maximum, "scope": "Shared GPU correctness; one resident generator; request includes temporary-store publication, excludes loading and close. Per-session counters are not whole-device usage. No isolated performance claim."})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("CURRENT_MEDIA %s", measurement)
		})
	}
	if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
		t.Fatal(err)
	}
	if err := generator.Reset(t.Context(), cases[0].request); err == nil {
		t.Fatal("closed generator accepted work")
	}
	t.Log("resource: task=SimpleDiffusion state=not_busy scope=owned-session close=completed")
}
