//go:build windows

package sensenovarecipe

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// Denominator and immutable artifacts: docs/image_video_protocol.json.
// Exact output equality retains the original reviews, including smoke defects.
func TestFrozenSenseNovaGeneration(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_SENSENOVA_FULL") != "1" {
		t.Skip("set OVERGO_SENSENOVA_FULL=1 for the frozen fifty-step case")
	}
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	referenceStore, err := overgodb.OpenReadOnly(cmp.Or(os.Getenv("OVERGO_SMOKE_REFERENCE_STORE"), roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := referenceStore.Close(); err != nil {
			t.Error(err)
		}
	})
	directory := testutil.ModelArtifactDir(t, "SenseNova-U1-8B-MoT-Infographic-V3")
	inventory, err := modelartifact.FromHFPath(directory)
	if err != nil {
		t.Fatal(err)
	}
	if inventory.Manifest.ID.String() != "model:sha256:03a237541f9d21f2c0fd477e5f29b56fecaf5dfa26a8cadc8da69aa5e3a42476" {
		t.Fatal("frozen model changed", inventory.Manifest.ID)
	}
	profile, err := routedlm.InspectFlowProfile(directory)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, inventory.Manifest.ID, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	if definition.ID.String() != "recipe:sha256:5a729e98ef825033d4073d0d0046ef3e63cfb6a519f8076bda01a29d151569ee" {
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
	profileContent, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Contents = append(batch.Contents, profileContent)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	generator, err := LoadGenerator(t.Context(), store, directory, definition)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Error(err)
		}
	})
	var library *driver.Library
	if err := generator.worker.Do(t.Context(), func(state *device.State) error { library = state.Driver; return nil }); err != nil {
		t.Fatal(err)
	}
	executor, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	if err := RegisterRuntime(executor, inventory.Manifest.ID, generator); err != nil {
		t.Fatal(err)
	}
	for index, tc := range []struct{ input, output string }{
		{"818cf1f980c17a498039c0efa4f007e630877424592b5c2013255c36f0223f43", "e652ebc738f538475c2ce188ac4b2f4daca3e00fd0f9b1ee0b322d603c9c9a7d"},
		{"f3caa7db1e6aef43ec9a0caddff9971ab23e94a6e9a8b316eea6d2debcc59e10", "5f3a56dff6b7a4a9e96d3d129082fdc349717357ff48d54454ff9535fb0cce68"},
		{"44ae255ebc22b8ab3adea63bde557ce5b85b6020aa993a015c411dd06b18fc42", "e8bf6ac4f2f33364f1852d807d7953acbe348c196cc76943552a8963a1811637"},
	} {
		t.Run(fmt.Sprintf("case-%d", index+1), func(t *testing.T) {
			inputID, err := artifact.ParseID("file:sha256:" + tc.input)
			if err != nil {
				t.Fatal(err)
			}
			input, present, err := artifact.ReadContent(t.Context(), referenceStore, inputID)
			if err != nil || !present {
				t.Fatalf("required frozen input %s: present=%v error=%v", inputID, present, err)
			}
			if err := input.Validate(); err != nil {
				t.Fatal(err)
			}
			var request GenerationRequest
			if err := strictjson.DecodeBytes(input.Data, &request); err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(filepath.Join(root, "docs/media_samples", tc.output+".png"))
			if err != nil {
				t.Fatal(err)
			}
			operation, err := workflowruntime.ExecutionID(definition.ID, t.Name())
			if err != nil {
				t.Fatal(err)
			}
			before := library.ExecutionStats()
			var hostBefore, hostAfter runtime.MemStats
			runtime.ReadMemStats(&hostBefore)
			started := time.Now()
			result, err := executor.ExecuteProgram(t.Context(), t.Name(), operation, nil, program, map[recipe.PortName]workflowruntime.Value{definition.Inputs[0].Name: workflowruntime.ArtifactValue(definition.Inputs[0].Data, request, input)})
			wall := time.Since(started)
			runtime.ReadMemStats(&hostAfter)
			if err != nil {
				t.Fatal(err)
			}
			if result.Run.Outcome != runrecord.OutcomeSucceeded || result.Run.Recipe != definition.ID || len(result.Run.Inputs) != 1 || result.Run.Inputs[0] != inputID || len(result.Run.Outputs) != 1 || result.Run.Outputs[0].String() != "output:sha256:"+tc.output {
				t.Fatalf("frozen lineage changed: %+v", result.Run)
			}
			value, ok := result.Outputs[definition.Outputs[0].Name].Single()
			if !ok {
				t.Fatal("missing scalar image")
			}
			image, ok := value.Value.(latentimage.EncodedImage)
			if !ok || image.Width != request.Width || image.Height != request.Height || !bytes.Equal(image.Data, want) {
				t.Fatal("frozen pixels or dimensions changed")
			}
			// The shared PNG encoder rejects non-finite floats before quantization.
			after := library.ExecutionStats()
			if after.KernelLaunches <= before.KernelLaunches || after.DeviceToHostBytes <= before.DeviceToHostBytes {
				t.Fatal("request performed no GPU work")
			}
			row, err := json.Marshal(map[string]any{"case": t.Name(), "model": inventory.Manifest.ID, "recipe": definition.ID, "input": inputID, "output": result.Run.Outputs[0], "run": result.Run.ID, "request": request, "request_ns": wall.Nanoseconds(), "allocated_bytes": hostAfter.TotalAlloc - hostBefore.TotalAlloc, "before": before, "after": after, "owned_memory": library.MemoryStats(), "phases": result.NodeWalls, "minimum": image.Minimum, "maximum": image.Maximum, "scope": "Shared GPU correctness, one loaded generator. Request includes temporary-store publication, excludes inventory/load/close. Owned GPU counters and process allocation deltas are not whole-device metrics. No matched speedup claim."})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("CURRENT_MEDIA %s", row)
		})
	}
	if err := generator.Close(context.WithoutCancel(t.Context())); err != nil {
		t.Fatal(err)
	}
	if stats := library.MemoryStats(); stats.CurrentBytes != 0 || stats.Allocations != 0 {
		t.Fatalf("generator leaked owned allocations: %+v", stats)
	}
	t.Log("resource: task=SenseNova state=not_busy scope=owned-session current_bytes=0")
}
