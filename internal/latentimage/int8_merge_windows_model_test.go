//go:build modeltest && windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// loraMergeCapture is what ComfyUI made of the MOSS Ultra workflow's int8 ConvRot projections with an adapter merged in
// while a section ran, as scripts/python_golden_references/comfy_lora_merge_capture.py captured it and
// comfy_lora_merge_golden.py wrote it: for each weight of the graph's (a fused qkv as its q, k and v thirds), the seed
// its module's rounding took and the SHA-256 of its int8 rows and of its F32 scales.
type loraMergeCapture struct {
	Section  string  `json:"section"`
	Adapter  string  `json:"adapter"`
	Strength float32 `json:"strength"`
	Weights  []struct {
		Weight string `json:"weight"`
		Module string `json:"module"`
		Seed   uint64 `json:"seed"`
		Rows   uint32 `json:"rows"`
		Inner  uint32 `json:"inner"`
		Int8   string `json:"int8_sha256"`
		Scales string `json:"scales_sha256"`
	} `json:"weights"`
}

// TestMOSSAdapterSectionsMultiplyAsComfyUI holds MOSS's sections with adapters, which patch its joint projections, to
// multiplying them as ComfyUI does (moss-int8-adapters): an adapter merged into an int8 ConvRot projection is ComfyUI's
// merge, bit for bit (holdLoRAMerge), and each section with adapters drawn from a picture (depth control and the Turbo
// sections in turboCases), sampling its projections in int8 with its adapters merged in, keeps at least the agreement
// with its recording its final latent keeps with them in BF16 beside their low-rank branches. Read 2026-11-19: depth
// control's latent 16.92 dB merged against 14.37 in BF16, image to image Turbo's 14.00 against 6.95, depth control
// Turbo's 5.49 against 2.08.
func TestMOSSAdapterSectionsMultiplyAsComfyUI(t *testing.T) {
	cudatest.Require(t)
	root := ultraModelRoot(t)
	ultraDepth(t, root)
	t.Run("merge", func(t *testing.T) { holdLoRAMerge(t, root) })
	t.Run("sections", func(t *testing.T) {
		var cases []goldenCase
		for _, golden := range goldenCases {
			if golden.name == "depth control" {
				cases = append(cases, golden)
			}
		}
		holdMergedSections(t, root, append(cases, turboCases...))
	})
}

// holdMergedSections reads each section's agreement with its recording with BF16 projections and then with int8
// projections and its adapters merged in, in one run, and holds each section's final latent to keeping at least BF16's.
func holdMergedSections(t *testing.T, root string, cases []goldenCase) {
	profile := testcheck.Must(ResolveProfile(root))(t)
	served := samplingInt8
	t.Cleanup(func() { samplingInt8 = served })
	samplingInt8 = false
	bf16 := readGoldenCases(t, root, profile, cases)
	samplingInt8 = true
	merged := readGoldenCases(t, root, profile, cases)
	for _, golden := range cases {
		before, after := bf16[golden.name], merged[golden.name]
		t.Logf("%s: the final latent keeps %.2f dB with its adapters merged into int8 projections, %.2f dB in BF16; the finished picture %.2f dB and %.2f dB",
			golden.name, after.latent, before.latent, after.picture, before.picture)
		if !(after.latent >= before.latent) {
			t.Errorf("%s keeps %.2f dB of its recording's latent with its adapters merged into int8 projections, %.2f dB in BF16; want the merge's at least BF16's",
				golden.name, after.latent, before.latent)
		}
	}
}

// holdLoRAMerge holds the merge of an adapter into an int8 ConvRot projection (convRotMerge, executor.MergeInt8ConvRot)
// to ComfyUI's, as the depth control section merged moss-control into layer 0's projections
// (fixtures/comfy/moss-ultra/lora_merge.json): each weight's merge takes its module's seed, and its int8 rows and F32
// scales are ComfyUI's, bit for bit.
func holdLoRAMerge(t *testing.T, root string) {
	ctx := t.Context()
	var capture loraMergeCapture
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(testutil.FixturePath(t, "comfy", "moss-ultra", "lora_merge.json")))(t), &capture))
	if len(capture.Weights) == 0 {
		t.Fatal("the capture holds no merged weight")
	}
	adapter := testcheck.Must(readMediaAdapter(testcheck.Must(adapterPath(root, capture.Adapter))(t)))(t)
	source := testcheck.Must(safetensors.OpenSource(filepath.Join(root, "transformer")))(t)
	defer func() { testcheck.NoError(t, source.Close()) }()
	worker := testcheck.Must(device.New(0))(t)
	defer func() { testcheck.NoError(t, worker.Close()) }()
	runtime := testcheck.Must(executor.NewWithWorker(worker))(t)
	defer func() { testcheck.NoError(t, runtime.Close()) }()
	placed := map[string]driver.DevicePtr{}
	defer func() {
		testcheck.NoError(t, worker.Do(context.WithoutCancel(ctx), func(state *device.State) (err error) {
			for _, pointer := range placed {
				err = errors.Join(err, state.Driver.MemFree(pointer))
			}
			return err
		}))
	}()
	place := func(adapter *mediaAdapter, name string, _, _ uint64) (driver.DevicePtr, error) {
		if pointer, held := placed[name]; held {
			return pointer, nil
		}
		data, err := adapter.bf16Tensor(name)
		if err != nil {
			return 0, err
		}
		err = worker.Do(ctx, func(state *device.State) error {
			pointer, err := state.Driver.MemAlloc(uint64(len(data)))
			if err != nil {
				return err
			}
			placed[name] = pointer
			return state.Driver.MemcpyHtoD(pointer, data)
		})
		return placed[name], err
	}
	defer func() { testcheck.NoError(t, adapter.release()) }()
	for _, captured := range capture.Weights {
		t.Run(captured.Weight, func(t *testing.T) {
			patch, held := adapter.lowRank[strings.TrimSuffix(captured.Weight, ".weight")]
			if !held {
				t.Fatalf("%s patches no %s", capture.Adapter, captured.Weight)
			}
			merge := testcheck.Must(convRotMerge(source, captured.Weight, []adapterPair{{adapter: adapter, patch: patch, strength: capture.Strength}}, place, 0))(t)
			if merge.Seed != captured.Seed || merge.Rows != captured.Rows || merge.Inner != captured.Inner {
				t.Fatalf("the merge of %d x %d is seeded %d; ComfyUI's %s of %d x %d was seeded %d",
					merge.Rows, merge.Inner, merge.Seed, captured.Module, captured.Rows, captured.Inner, captured.Seed)
			}
			elements := uint64(merge.Rows) * uint64(merge.Inner)
			merged := make([]byte, elements+4*uint64(merge.Rows))
			testcheck.NoError(t, worker.Do(ctx, func(state *device.State) (err error) {
				output, err := state.Driver.MemAlloc(uint64(len(merged)))
				if err != nil {
					return err
				}
				defer func() { err = errors.Join(err, state.Driver.MemFree(output)) }()
				merge.Output = output
				if err := runtime.MergeInt8ConvRot(state, merge); err != nil {
					return err
				}
				return state.Driver.MemcpyDtoH(merged, output)
			}))
			if got := sha256Hex(merged[:elements]); got != captured.Int8 {
				t.Errorf("the merged int8 rows hash %s; ComfyUI's %s", got, captured.Int8)
			}
			if got := sha256Hex(merged[elements:]); got != captured.Scales {
				t.Errorf("the merged scales hash %s; ComfyUI's %s", got, captured.Scales)
			}
		})
	}
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
