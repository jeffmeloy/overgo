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

func TestMOSSRotaryJointCUDAGolden(t *testing.T) {
	cudatest.Require(t)
	oracle := jointFixture(t)
	spec := refinementSpec()
	spec.Layers = 2
	runtime, err := graphruntime.NewResidentSession(0)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close(context.WithoutCancel(t.Context()))
	for _, length := range []int{2, 4} {
		for block := range 2 {
			t.Run(fmt.Sprintf("%d/%d", length, block), func(t *testing.T) {
				p, err := compileRotaryJointProgram(spec, 1e-6, length+6, block, dtype.BF16)
				if err != nil {
					t.Fatal(err)
				}
				prefix := fmt.Sprintf("cuda-bf16.%d.block.%d.", length, block)
				program, err := runtime.Compile(t.Context(), "joint block", nil, p.output)
				if err != nil {
					t.Fatal(err)
				}
				defer runtime.Release(context.WithoutCancel(t.Context()), program)
				feeds := jointFeeds(t, p, oracle, prefix, dtype.BF16)
				for _, node := range p.weightInputs {
					values := feeds[node].Data
					err := runtime.Bind(t.Context(), program, "joint fixture", node.Name, node, func() ([]byte, error) {
						if node.Type == dtype.F32 {
							return driver.Bytes(values), nil
						}
						return quant.Quantize(node.Type, values)
					})
					if err != nil {
						t.Fatal(err)
					}
					delete(feeds, node)
				}
				got, err := runtime.Execute(t.Context(), program, feeds)
				if err != nil {
					t.Fatal(err)
				}
				compareRefinement(t, got[p.output].Data, refinementValues(t, oracle, prefix+"output"), .016)
				repeat, err := runtime.Execute(t.Context(), program, feeds)
				if err != nil {
					t.Fatal(err)
				}
				for i, x := range got[p.output].Data {
					if repeat[p.output].Data[i] != x {
						t.Fatal("joint block did not repeat exactly")
					}
				}
			})
		}
	}
}

func TestMOSSRotaryJointRMSBoundary(t *testing.T) {
	cudatest.Require(t)
	mossFixtureBytes(t, "joint-rms.safetensors", "8dee4bd3e3f65dfc431e974444f2fa081e176af8930602c6c08c4b48b2bf48ce")
	source, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "joint-rms.safetensors"))
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
	x := b.Input("x", dtype.F32, tensor.MustShape(3072))
	y := b.BF16Round(b.RMSNorm(x, 1e-6))
	program, err := runtime.Compile(t.Context(), "native joint RMS", nil, y)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Release(context.WithoutCancel(t.Context()), program)
	got, err := runtime.Execute(t.Context(), program, map[*tensor.Tensor]reference.Value{x: {Shape: x.Shape, Data: refinementValues(t, source, "input")}})
	if err != nil {
		t.Fatal(err)
	}
	want := refinementValues(t, source, "rounded")
	for i, value := range got[y].Data {
		if value != want[i] {
			t.Errorf("native RMS boundary %d got=%g want=%g", i, value, want[i])
		}
	}
}
