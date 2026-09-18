package oscillatorimage

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// Frozen cases: image_video_protocol.json, routed_images.json and
// conditioned_videos.json. Corrected GIFs preserve pixels and cumulative timing.
// This is CPU smoke acceptance; independent frames establish no motion quality.
func TestFrozenUn0Modalities(t *testing.T) {
	directory := artifactDir(t)
	for _, path := range []string{filepath.Join(directory, "config.json"), filepath.Join(directory, "model.safetensors"), referenceFixturePath(t, "un0_dynamics_golden.json"), referenceFixturePath(t, "un0_generator_golden.json")} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("required acceptance fixture %s: %v", path, err)
		}
	}
	inventory, err := modelartifact.FromFiles(directory, []modelartifact.FileSpec{
		{Path: filepath.Join(directory, "config.json"), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(directory, "model.safetensors"), Name: "weights", Role: artifact.ComponentWeights},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := inventory.Manifest.ID.String(); got != "model:sha256:04097c8ba3345cdb75cb258c4424643da57dd649cb0b46fb1bad63640303bb81" {
		t.Fatalf("frozen model changed: %s", got)
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
	model := loadArtifactModel(t)
	type frozenCase struct {
		request               any
		input, output, sample string
	}
	for _, group := range []struct {
		name     string
		prepare  recipe.ModuleID
		recipeID string
		cases    []frozenCase
	}{
		{"image", modelrecipe.ModuleOscillatorImagePrepare, "recipe:sha256:213d890a8988e995e520dda1178fcbcc446e006f69b01316bde61300b11c1415", []frozenCase{
			{Request{Class: 1, Seed: 111}, "8e601d036e88e152f15809ac993be461f9835d6241e54d851a42946195f3fbfb", "cf5dcb195503d8c9ce318a0fc8997441466c0aa3f911ae772f0ab82c13448e28", "cf5dcb195503d8c9ce318a0fc8997441466c0aa3f911ae772f0ab82c13448e28.png"},
			{Request{Class: 1, Seed: 11}, "82b9edbba80bbd74c5f6e3df94b3a7af504d2ffce8e690e91cca7da6a454172a", "a8bc7f12cb9a537b40bf150493a002251bbbc59620a69944e19778dd463da6f3", "a8bc7f12cb9a537b40bf150493a002251bbbc59620a69944e19778dd463da6f3.png"},
		}},
		{"video", modelrecipe.ModuleOscillatorVideoPrepare, "recipe:sha256:4f6267a0ddb44c1357c11ec4229706cabdeabdd7d946fa9cfc90a9b9a9448e9e", []frozenCase{
			{VideoRequest{Class: 1, Seed: 11, Frames: 12, Scale: 2}, "1d74176e310fa363bf987866df9d59bf942e301d2d326dbd501ac2b8bc01e80d", "8998678c48f16a2d4e73a63e764c030acf022775ea8744836cdd6b1ec2cd4024", "bbae395b6153b0f2581e72ba32f9c4b77c120ec04d47c8a127efb78fa8ed7448.gif"},
			{VideoRequest{Class: 1, Seed: 111, Frames: 12, Scale: 2}, "0bf08dc9e7fa42a77c7a375466cf0809af3cfc223fe31a027b8efdd064d6e48e", "5ce8cb116444a2bfbafbe9c8700a898c2e98ae1a20b7b6ca6ba7bcde0b954aee", "4869d07c17d398a6c5081b00c96a5a99402fbd4504980e659d81d68c6ef7c658.gif"},
		}},
	} {
		definition, err := modelrecipe.GenerationDefinition(group.prepare, inventory.Manifest.ID, artifact.ID{})
		if err != nil {
			t.Fatal(err)
		}
		if definition.ID.String() != group.recipeID {
			t.Fatalf("%s recipe changed: %s", group.name, definition.ID)
		}
		program, err := modelrecipe.CompileCapability(definition)
		if err != nil {
			t.Fatal(err)
		}
		executor, err := workflowruntime.NewForProgram(store, program)
		if err != nil {
			t.Fatal(err)
		}
		if err := RegisterRuntime(executor, inventory.Manifest.ID, model); err != nil {
			t.Fatal(err)
		}
		for index, tc := range group.cases {
			t.Run(fmt.Sprintf("%s/%d", group.name, index+1), func(t *testing.T) {
				input, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo."+group.name+"-gen-input.v1"), tc.request)
				if err != nil {
					t.Fatal(err)
				}
				if input.Descriptor.ID.String() != "file:sha256:"+tc.input {
					t.Fatal("frozen input changed")
				}
				want, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs/media_samples", tc.sample))
				if err != nil {
					t.Fatal(err)
				}
				operation, err := workflowruntime.ExecutionID(definition.ID, t.Name())
				if err != nil {
					t.Fatal(err)
				}
				var before, after runtime.MemStats
				runtime.ReadMemStats(&before)
				started := time.Now()
				result, err := executor.ExecuteProgram(t.Context(), t.Name(), operation, nil, program, map[recipe.PortName]workflowruntime.Value{
					definition.Inputs[0].Name: workflowruntime.ArtifactValue(recipe.DataClassConditioning, tc.request, input),
				})
				wall := time.Since(started)
				runtime.ReadMemStats(&after)
				if err != nil {
					t.Fatal(err)
				}
				if result.Run.Outcome != runrecord.OutcomeSucceeded || len(result.Run.Outputs) != 1 || result.Run.Outputs[0].String() != "output:sha256:"+tc.output {
					t.Fatalf("frozen output/lineage changed: %+v", result.Run)
				}
				value, ok := result.Outputs[definition.Outputs[0].Name].Single()
				if !ok {
					t.Fatal("missing scalar output")
				}
				var data []byte
				switch output := value.Value.(type) {
				case latentimage.EncodedImage:
					data = output.Data
				case EncodedVideo:
					data = output.Data
				default:
					t.Fatalf("unexpected output %T", output)
				}
				if !bytes.Equal(data, want) {
					t.Fatal("encoded pixels or timing changed")
				}
				// Inspect floats before quantization; equality of encoded bytes alone
				// cannot rule out a non-finite value concealed by the converter.
				request := Request{}
				frames := 1
				switch r := tc.request.(type) {
				case Request:
					request = r
				case VideoRequest:
					request = Request{Class: r.Class, Seed: r.Seed}
					frames = r.Frames
				}
				for frame := range frames {
					pixels, err := generatePlanar(t, model, Request{Class: request.Class, Seed: request.Seed + int64(frame)})
					if err != nil {
						t.Fatal(err)
					}
					for _, pixel := range pixels {
						if math.IsNaN(float64(pixel)) || math.IsInf(float64(pixel), 0) {
							t.Fatal("non-finite prequantization pixel")
						}
					}
				}
				measurement, err := json.Marshal(struct {
					Case                              string `json:"case"`
					Model, Recipe, Input, Output, Run artifact.ID
					WallNS                            int64                      `json:"wall_ns"`
					AllocatedBytes                    uint64                     `json:"allocated_bytes"`
					Stages                            []workflowruntime.NodeWall `json:"stages"`
					Scope                             string                     `json:"scope"`
				}{t.Name(), inventory.Manifest.ID, definition.ID, input.Descriptor.ID, result.Run.Outputs[0], result.Run.ID, wall.Nanoseconds(), after.TotalAlloc - before.TotalAlloc, result.NodeWalls, "CPU recipe execution and temporary-store publication; one shared model residency; no GPU allocation or isolated performance claim. Float inspection and native reference checks are separate."})
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("CURRENT_MEDIA %s", measurement)
			})
		}
	}
}
