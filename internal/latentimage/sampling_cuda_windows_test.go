//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
)

func samplingModelFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	directory := filepath.Join(root, "transformer")
	if e := os.Mkdir(directory, 0700); e != nil {
		t.Fatal(e)
	}
	p, e := compileRotaryDenoiserProgram(denoiserSpec(), 1e-6, 2, 3, 4, nil, dtype.BF16)
	if e != nil {
		t.Fatal(e)
	}
	feeds := denoiserFeeds(t, p, 0, dtype.BF16)
	values := map[string][]float32{}
	shapes := map[string][]int{}
	for _, node := range p.weightInputs {
		values[node.Name] = feeds[node].Data
		shape := node.Shape.Slice()
		dims := make([]int, len(shape))
		for i, n := range shape {
			dims[len(shape)-1-i] = int(n)
		}
		shapes[node.Name] = dims
	}
	if e := safetensors.Save(filepath.Join(directory, "weights.safetensors"), values, shapes, nil); e != nil {
		t.Fatal(e)
	}
	return root
}

func TestMOSSSamplingWeightFailureReleasesPartialBindings(t *testing.T) {
	cudatest.Require(t)
	runtime, e := graphruntime.NewResidentSession(0)
	if e != nil {
		t.Fatal(e)
	}
	defer runtime.Close(context.WithoutCancel(t.Context()))
	b := tensor.NewBuilder()
	first := b.Input("available", dtype.F32, tensor.MustShape(2))
	missing := b.Input("missing", dtype.F32, first.Shape)
	output := b.Add(first, missing)
	root := t.TempDir()
	if e := safetensors.Save(filepath.Join(root, "weights.safetensors"), map[string][]float32{"available": {1, 2}}, map[string][]int{"available": {2}}, nil); e != nil {
		t.Fatal(e)
	}
	if _, e := compileResidentWeights(t.Context(), runtime, "partial binding failure", root, tensor.WeightBindings{first, missing}, nil, output); e == nil {
		t.Fatal("missing weight accepted")
	}
	if runtime.StaticBytes() != 0 {
		t.Fatalf("partial binding retained %d bytes", runtime.StaticBytes())
	}
}

func TestMOSSSamplingCancellationRecovers(t *testing.T) {
	cudatest.Require(t)
	oracle, cases := samplingFixture(t)
	root := samplingModelFixture(t)
	runtime, e := graphruntime.NewResidentSession(0)
	if e != nil {
		t.Fatal(e)
	}
	defer runtime.Close(context.WithoutCancel(t.Context()))
	c := cases[0]
	settings := samplingSettings(t, c)
	prefix := "cuda-bf16." + c.Name + "."
	initial, e := torchrng.CPUNormal(42, 24)
	if e != nil {
		t.Fatal(e)
	}
	for _, boundary := range []string{"initial_latents", "prediction.0", "prediction.1", "velocity.0", "latents.0"} {
		t.Run(boundary, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			seen := false
			_, e := runRotarySampling(ctx, runtime, root, denoiserSpec(), 2, 3, initial, refinementValues(t, oracle, prefix+"positive"), refinementValues(t, oracle, prefix+"negative"), settings, func(name string, _ []float32) error {
				if name == boundary {
					seen = true
					cancel()
				}
				return ctx.Err()
			})
			if !seen || !errors.Is(e, context.Canceled) {
				t.Fatalf("cancellation reached=%v error=%v", seen, e)
			}
			if runtime.StaticBytes() != 0 {
				t.Fatalf("cancelled run retained weights: %d", runtime.StaticBytes())
			}
			result, e := runRotarySampling(t.Context(), runtime, root, denoiserSpec(), 2, 3, initial, refinementValues(t, oracle, prefix+"positive"), refinementValues(t, oracle, prefix+"negative"), settings, nil)
			if e != nil {
				t.Fatal(e)
			}
			value, e := result.outputs.CopyToHost(t.Context(), result.node)
			if e != nil {
				t.Fatal(e)
			}
			compareRefinement(t, value.Data, refinementValues(t, oracle, prefix+"output"), 1.3e-6)
			if e := result.outputs.Release(t.Context()); e != nil {
				t.Fatal(e)
			}
		})
	}
}

func TestMOSSSamplingCUDAGolden(t *testing.T) {
	cudatest.Require(t)
	oracle, cases := samplingFixture(t)
	root := samplingModelFixture(t)
	runtime, e := graphruntime.NewResidentSession(0)
	if e != nil {
		t.Fatal(e)
	}
	defer runtime.Close(context.WithoutCancel(t.Context()))
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			prefix := "cuda-bf16." + c.Name + "."
			settings := samplingSettings(t, c)
			initial, e := torchrng.CPUNormal(42, 24)
			if e != nil {
				t.Fatal(e)
			}
			hashes := map[string][sha256.Size]byte{}
			for pass := range 2 {
				result, e := runRotarySampling(t.Context(), runtime, root, denoiserSpec(), 2, 3, initial, refinementValues(t, oracle, prefix+"positive"), refinementValues(t, oracle, prefix+"negative"), settings, func(name string, values []float32) error {
					tolerance := 1.3e-6
					if strings.HasPrefix(name, "prediction.") {
						tolerance = .016
					}
					compareRefinement(t, values, refinementValues(t, oracle, prefix+samplingReferenceName(name)), tolerance)
					hash := sha256.Sum256(driver.Bytes(values))
					if pass == 0 {
						hashes[name] = hash
					} else if hashes[name] != hash {
						return errors.New("sampling repeat differs")
					}
					return nil
				})
				if e != nil {
					t.Fatal(e)
				}
				value, e := result.outputs.CopyToHost(t.Context(), result.node)
				if e != nil {
					t.Fatal(e)
				}
				compareRefinement(t, value.Data, refinementValues(t, oracle, prefix+"output"), 1.3e-6)
				if e := result.outputs.Release(t.Context()); e != nil {
					t.Fatal(e)
				}
				if runtime.StaticBytes() != 0 {
					t.Fatal("sampling weights survived execution")
				}
			}
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			if _, e := runRotarySampling(ctx, runtime, root, denoiserSpec(), 2, 3, initial, refinementValues(t, oracle, prefix+"positive"), nil, settings, nil); !errors.Is(e, context.Canceled) {
				t.Fatalf("cancelled run error=%v", e)
			}
		})
	}
}
