//go:build windows

package latentvideo

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// Inputs are frozen by image_video_protocol; accepted bundles correct GIF
// cadence and LiveEdit's signed pixel range without changing model arithmetic.
func TestFrozenLatentVideoGeneration(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_WAN_FULL") != "1" {
		t.Skip("set OVERGO_WAN_FULL=1 for the complete frozen 81-frame clip")
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
	for _, edit := range []bool{false, true} {
		name := "Wan2.1-T2V-1.3B"
		if edit {
			name = "LiveEdit"
		}
		t.Run(name, func(t *testing.T) {
			fixture := newFrozenVideoFixture(t, edit)
			cases := []struct {
				inputs []string
				output string
			}{
				{[]string{"548efce9b4a867734a22f75d4fdb1c00faa9582945fd355f853fdeb71de5ecc9"}, "31ee8f128572912a720e68296c646f3dc0e3e0e413ef48392872e713e6152240"},
				{[]string{"ddb180564d8a0a1fdacff9ce2f3276a3282a00b96af71957eda687646d7b8680"}, "c733846c0bcf7d0cfa24a43d06665f609f2ac399fe384e1cfe0ced8bb5c41b99"},
				{[]string{"364be3ce53202ce6b5be13d1265ff1e909f6c232b1796f06c73d451fad326a72"}, "d5c72f13e0bf07a1fb8268917b4da15f53e1343bfcdb3093a309591bb08f8ff9"},
			}
			if edit {
				cases = []struct {
					inputs []string
					output string
				}{
					{[]string{"13d22d58fe9d17432757def7ec1cdcb170f07358c6ccaf0a5d4dcfdeec1e95ca", "e14e51433c7ccd5111ee3e176a4da64342f7d24412979eb1faa406836ff39e91"}, "a7b319f0c798b2ed42bbd68e4376c81c6dec00d33f9f78a7051704ded8c7fa21"},
					{[]string{"35121e6f6d23f5d51b5438029d637dd3fb43d9efa3d59adc5f12bd7c0059bf78", "e14e51433c7ccd5111ee3e176a4da64342f7d24412979eb1faa406836ff39e91"}, "a7b319f0c798b2ed42bbd68e4376c81c6dec00d33f9f78a7051704ded8c7fa21"},
				}
			}
			var wan *WanRuntime
			var wanLibraries []*driver.Library
			t.Cleanup(func() {
				if wan != nil {
					closeFrozenVideo(t, wan.Close, wanLibraries)
				}
			})
			for index, tc := range cases {
				t.Run(fmt.Sprintf("case-%d", index+1), func(t *testing.T) {
					inputs := make([]artifact.Content, len(tc.inputs))
					for index, hash := range tc.inputs {
						id, err := artifact.ParseID("file:sha256:" + hash)
						if err != nil {
							t.Fatal(err)
						}
						content, present, err := artifact.ReadContent(t.Context(), reference, id)
						if err != nil || !present {
							t.Fatalf("required input %s: present=%v error=%v", id, present, err)
						}
						if err := content.Validate(); err != nil {
							t.Fatal(err)
						}
						inputs[index] = content
					}
					want, err := os.ReadFile(filepath.Join(root, "docs/media_samples", tc.output+".gif"))
					if err != nil {
						t.Fatal(err)
					}
					executor, err := workflowruntime.NewForProgram(fixture.store, fixture.program)
					if err != nil {
						t.Fatal(err)
					}
					values := make(map[recipe.PortName]workflowruntime.Value)
					var workers []*device.Worker
					var libraries []*driver.Library
					var frames, height, width int
					loaded := time.Now()
					if edit {
						var request ReferenceEditRequest
						if err := strictjson.DecodeBytes(inputs[0].Data, &request.Condition); err != nil {
							t.Fatal(err)
						}
						if err := strictjson.DecodeBytes(inputs[1].Data, &request.Source); err != nil {
							t.Fatal(err)
						}
						model, err := LoadLiveEditRuntime(t.Context(), fixture.store, fixture.directory, fixture.program, request)
						if err != nil {
							t.Fatal(err)
						}
						workers = []*device.Worker{model.runtime.encoder.codec.worker, model.runtime.denoiser.worker, model.runtime.decoder.worker}
						t.Cleanup(func() { closeFrozenVideo(t, model.Close, libraries) })
						if err := RegisterLiveEditRuntime(executor, fixture.model, model); err != nil {
							t.Fatal(err)
						}
						for index, value := range []any{request.Condition, request.Source} {
							port := fixture.definition.Inputs[index]
							values[port.Name] = workflowruntime.ArtifactValue(port.Data, value, inputs[index])
						}
						frames, height, width = request.Source.Frames, request.Source.Height, request.Source.Width
					} else {
						var request WanRequest
						if err := strictjson.DecodeBytes(inputs[0].Data, &request); err != nil {
							t.Fatal(err)
						}
						if wan != nil && wan.Reset(t.Context(), request) != nil {
							closeFrozenVideo(t, wan.Close, wanLibraries)
							wan = nil
							wanLibraries = nil
						}
						if wan == nil {
							wan, err = LoadWanRuntime(t.Context(), fixture.store, fixture.directory, fixture.program, request)
							if err != nil {
								t.Fatal(err)
							}
						}
						workers = []*device.Worker{wan.generator.denoiser.worker, wan.generator.decoder.worker}
						if err := RegisterWanRuntime(executor, fixture.model, wan); err != nil {
							t.Fatal(err)
						}
						port := fixture.definition.Inputs[0]
						values[port.Name] = workflowruntime.ArtifactValue(port.Data, request, inputs[0])
						frames, height, width = request.Frames, request.Height, request.Width
					}
					loadWall := time.Since(loaded)
					for _, worker := range workers {
						if err := worker.Do(t.Context(), func(state *device.State) error { libraries = append(libraries, state.Driver); return nil }); err != nil {
							t.Fatal(err)
						}
					}
					if !edit {
						wanLibraries = libraries
					}
					before := make([]driver.ExecutionStats, len(libraries))
					for index, library := range libraries {
						before[index] = library.ExecutionStats()
					}
					operation, err := workflowruntime.ExecutionID(fixture.definition.ID, t.Name())
					if err != nil {
						t.Fatal(err)
					}
					var hostBefore, hostAfter runtime.MemStats
					runtime.ReadMemStats(&hostBefore)
					started := time.Now()
					result, err := executor.ExecuteProgram(t.Context(), t.Name(), operation, nil, fixture.program, values)
					wall := time.Since(started)
					runtime.ReadMemStats(&hostAfter)
					if err != nil {
						t.Fatal(err)
					}
					wantInputs := make([]artifact.ID, len(inputs))
					for index, input := range inputs {
						wantInputs[index] = input.Descriptor.ID
					}
					if result.Run.Outcome != runrecord.OutcomeSucceeded || result.Run.Recipe != fixture.definition.ID || !slices.Equal(result.Run.Inputs, wantInputs) || len(result.Run.Outputs) != 1 || result.Run.Outputs[0].String() != "output:sha256:"+tc.output {
						t.Fatalf("frozen lineage changed: %+v", result.Run)
					}
					value, ok := result.Outputs[fixture.definition.Outputs[0].Name].Single()
					if !ok {
						t.Fatal("missing scalar video")
					}
					video, ok := value.Value.(EncodedVideo)
					if !ok || video.Frames != frames || video.Height != height || video.Width != width || video.FPS != fixture.profile.SampleFPS || !bytes.Equal(video.Data, want) {
						t.Fatal("frozen frame pixels, geometry or cadence changed")
					}
					// GIF encoding rejects non-finite floats before quantization.
					after := make([]driver.ExecutionStats, len(libraries))
					memory := make([]driver.MemoryStats, len(libraries))
					for index, library := range libraries {
						after[index], memory[index] = library.ExecutionStats(), library.MemoryStats()
						if after[index].KernelLaunches+after[index].GraphLaunches <= before[index].KernelLaunches+before[index].GraphLaunches {
							t.Fatalf("owner %d performed no GPU work", index)
						}
					}
					row, err := json.Marshal(map[string]any{"case": t.Name(), "recipe": fixture.definition.ID, "inputs": wantInputs, "output": result.Run.Outputs[0], "run": result.Run.ID, "frames": frames, "height": height, "width": width, "load_ns": loadWall.Nanoseconds(), "request_ns": wall.Nanoseconds(), "allocated_bytes": hostAfter.TotalAlloc - hostBefore.TotalAlloc, "before": before, "after": after, "owned_memory": memory, "phases": result.NodeWalls, "scope": "Shared GPU correctness. Request includes temporary-store publication, excludes load/close. Owned counters and process allocation deltas are not whole-device measurements. Exact corrected samples preserve original visual limitations."})
					if err != nil {
						t.Fatal(err)
					}
					t.Logf("CURRENT_MEDIA %s", row)
				})
			}
		})
	}
}

type frozenVideoFixture struct {
	model      artifact.ID
	store      *overgodb.Store
	directory  string
	definition recipe.Definition
	program    recipe.Program
	profile    Profile
}

func newFrozenVideoFixture(t *testing.T, edit bool) frozenVideoFixture {
	t.Helper()
	wan := testutil.ModelArtifactDir(t, "Wan2.1-T2V-1.3B")
	directory, inventoryRoot := wan, wan
	files := []modelartifact.FileSpec{
		{Path: filepath.Join(wan, "config.json"), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(wan, "diffusion_pytorch_model.safetensors"), Name: "denoiser/weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(wan, "Wan2.1_VAE.pth"), Name: "vae/weights", Role: artifact.ComponentWeights},
	}
	wantModel, wantRecipe := "7bb7f43a42450222e958436862c219aa87806f0301c457a0d32043da4b46cce1", "bff12bb8946afe0d8ca84d635e0ff275855c94a22a99df91e03d319c32ed54b4"
	if edit {
		directory = testutil.ModelArtifactDir(t, "LiveEdit")
		inventoryRoot = filepath.Dir(directory)
		files = []modelartifact.FileSpec{
			{Path: filepath.Join(directory, "ar-forcing_002000.pt"), Name: "liveedit/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "config.json"), Name: "wan/config", Role: artifact.ComponentConfig},
			{Path: filepath.Join(wan, "diffusion_pytorch_model.safetensors"), Name: "wan/weights", Role: artifact.ComponentWeights},
			{Path: filepath.Join(wan, "Wan2.1_VAE.pth"), Name: "wan/vae", Role: artifact.ComponentWeights},
		}
		wantModel, wantRecipe = "67c7b0c56742b6f7477ec2c414ae8a337ace22e311b4f03f55860916706314df", "7cd66ed08ebee5e566dca3682e7d6d9f0b06f8384f19b1f1dfb416a272126965"
	}
	inventory, err := modelartifact.FromFiles(inventoryRoot, files)
	if err != nil {
		t.Fatal(err)
	}
	if inventory.Manifest.ID.String() != "model:sha256:"+wantModel {
		t.Fatal("frozen model changed", inventory.Manifest.ID)
	}
	profile, err := ResolveProfile(wan)
	if err != nil {
		t.Fatal(err)
	}
	var definition recipe.Definition
	if edit {
		definition, err = modelrecipe.ReferenceVideoEditDefinition(inventory.Manifest.ID, profile.ID)
	} else {
		definition, err = modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentVideoPrepare, inventory.Manifest.ID, profile.ID)
	}
	if err != nil {
		t.Fatal(err)
	}
	if definition.ID.String() != "recipe:sha256:"+wantRecipe {
		t.Fatal("frozen recipe/profile changed", definition.ID)
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
	content, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Contents = append(batch.Contents, content)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return frozenVideoFixture{model: inventory.Manifest.ID, store: store, directory: directory, definition: definition, program: program, profile: profile}
}

func closeFrozenVideo(t *testing.T, close func(context.Context) error, libraries []*driver.Library) {
	t.Helper()
	if err := close(context.WithoutCancel(t.Context())); err != nil {
		t.Fatal(err)
	}
	for index, library := range libraries {
		if stats := library.MemoryStats(); stats.CurrentBytes != 0 || stats.Allocations != 0 {
			t.Fatalf("closed owner %d leaked: %+v", index, stats)
		}
	}
	t.Log("resource: task=latent-video state=not_busy scope=owned-session current_bytes=0")
}
