//go:build windows

package latentimage

import (
	"context"
	"fmt"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/quant"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"testing"
)

func TestMOSSRotaryRefinementCUDAGolden(t *testing.T) {
	cudatest.Require(t)
	oracle := refinementFixture(t)
	runtime, e := graphruntime.NewResidentSession(0)
	if e != nil {
		t.Fatal(e)
	}
	defer runtime.Close(context.WithoutCancel(t.Context()))
	for _, length := range []int{2, 4} {
		t.Run(fmt.Sprint(length), func(t *testing.T) {
			p, e := compileRotaryRefinementProgram(refinementSpec(), 1e-6, length, dtype.BF16)
			if e != nil {
				t.Fatal(e)
			}
			prefix := fmt.Sprintf("cuda-bf16.%d.", length)
			outputs := p.RefinementOutputs
			program, e := runtime.Compile(t.Context(), "rotary refinement", nil, outputs...)
			if e != nil {
				t.Fatal(e)
			}
			defer runtime.Release(context.WithoutCancel(t.Context()), program)
			feeds := refinementFeeds(t, p, oracle, prefix, dtype.BF16)
			for _, node := range p.weightInputs {
				values := feeds[node].Data
				e = runtime.Bind(t.Context(), program, "refinement fixture", node.Name, node, func() ([]byte, error) {
					if node.Type == dtype.F32 {
						return driver.Bytes(values), nil
					}
					return quant.Quantize(node.Type, values)
				})
				if e != nil {
					t.Fatal(e)
				}
				delete(feeds, node)
			}
			got, e := runtime.Execute(t.Context(), program, feeds)
			if e != nil {
				t.Fatal(e)
			}
			for i, n := range p.RefinementOutputs {
				compareRefinement(t, got[n].Data, refinementValues(t, oracle, fmt.Sprintf("%sblock.%d", prefix, i)), .016)
			}
			compareRefinement(t, got[p.Fused].Data, refinementValues(t, oracle, prefix+"output"), .016)
		})
	}
}

func TestMOSSRotaryRefinementRMSBoundary(t *testing.T) {
	cudatest.Require(t)
	mossFixtureBytes(t, "refinement-rms.safetensors", "1764b08ccfe0ff3c1509e644cff112e89a0922dba27f511031a51fa414e24436")
	source, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "refinement-rms.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	runtime, err := graphruntime.NewResidentSession(0)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close(context.WithoutCancel(t.Context()))
	b := tensor.NewBuilder()
	x := b.Input("x", dtype.F32, tensor.MustShape(128))
	y := b.BF16Round(b.RMSNorm(x, 1e-6))
	program, err := runtime.Compile(t.Context(), "native RMS boundary", nil, y)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Release(context.WithoutCancel(t.Context()), program)
	got, err := runtime.Execute(t.Context(), program, map[*tensor.Tensor]reference.Value{x: {Shape: x.Shape, Data: refinementValues(t, source, "input")}})
	if err != nil {
		t.Fatal(err)
	}
	want := refinementValues(t, source, "raw")
	dtype.RoundBF16Slice(want)
	for i, value := range got[y].Data {
		if value != want[i] {
			t.Fatalf("native RMS boundary %d got=%g want=%g", i, value, want[i])
		}
	}
}
