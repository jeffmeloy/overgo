//go:build windows

package latentimage

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/quant"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

func TestMOSSRotaryDenoiserCUDAGolden(t *testing.T) {
	cudatest.Require(t)
	s := denoiserFixture(t)
	runtime, err := graphruntime.NewResidentSession(0)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close(context.WithoutCancel(t.Context()))
	for index, length := range []int{4, 2} {
		for _, variant := range []struct {
			name string
			skip []int
		}{{"all", nil}, {"skip-1", []int{1}}, {"skip-all", []int{0, 1}}} {
			t.Run(fmt.Sprintf("%d/%s", length, variant.name), func(t *testing.T) {
				p, err := compileRotaryDenoiserProgram(denoiserSpec(), 1e-6, 2, 3, length, variant.skip, dtype.BF16)
				if err != nil {
					t.Fatal(err)
				}
				outputs, named := denoiserOutputs(p)
				program, err := runtime.Compile(t.Context(), "native complete denoiser", nil, outputs...)
				if err != nil {
					t.Fatal(err)
				}
				defer runtime.Release(context.WithoutCancel(t.Context()), program)
				feeds := denoiserFeeds(t, p, index, dtype.BF16)
				for _, node := range p.weightInputs {
					values := feeds[node].Data
					if err := runtime.Bind(t.Context(), program, "denoiser fixture", node.Name, node, func() ([]byte, error) {
						if node.Type == dtype.F32 {
							return driver.Bytes(values), nil
						}
						return quant.Quantize(node.Type, values)
					}); err != nil {
						t.Fatal(err)
					}
					delete(feeds, node)
				}
				got, err := runtime.Execute(t.Context(), program, feeds)
				if err != nil {
					t.Fatal(err)
				}
				repeat, err := runtime.Execute(t.Context(), program, feeds)
				if err != nil {
					t.Fatal(err)
				}
				prefix := fmt.Sprintf("cuda-bf16.%d.%s.", length, variant.name)
				for name, node := range named {
					t.Run(name, func(t *testing.T) {
						compareRefinement(t, got[node].Data, refinementValues(t, s, prefix+name), .016)
						for i, x := range got[node].Data {
							if repeat[node].Data[i] != x {
								t.Fatal("forward did not repeat exactly")
							}
						}
					})
				}
			})
		}
	}
}

func TestMOSSRotaryDenoiserFinalOutputResidency(t *testing.T) {
	cudatest.Require(t)
	oracle := denoiserFixture(t)
	p, err := compileRotaryDenoiserProgram(denoiserSpec(), 1e-6, 2, 3, 4, []int{0, 1, 1}, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	feeds := denoiserFeeds(t, p, 0, dtype.BF16)
	values := map[string][]float32{}
	shapes := map[string][]int{}
	for _, node := range p.weightInputs {
		values[node.Name] = feeds[node].Data
		shape := node.Shape.Slice()
		dimensions := make([]int, len(shape))
		for i, n := range shape {
			dimensions[len(shape)-1-i] = int(n)
		}
		shapes[node.Name] = dimensions
		delete(feeds, node)
	}
	root := t.TempDir()
	if err := safetensors.Save(filepath.Join(root, "weights.safetensors"), values, shapes, nil); err != nil {
		t.Fatal(err)
	}
	f := newResidentFixture(t, t.Context(), "final-only skipped denoiser", root, p.weightInputs, p.output)
	got := f.execute(t, t.Context(), feeds)
	compareRefinement(t, got[p.output].Data, refinementValues(t, oracle, "cuda-bf16.4.skip-all.output"), .016)
	var expected uint64
	for _, node := range p.weightInputs {
		if _, ok := f.graph.Graph.InputSlot(node); ok {
			n, err := node.Shape.Bytes(node.Type)
			if err != nil {
				t.Fatal(err)
			}
			expected += n
		}
	}
	if f.runtime.StaticBytes() != expected {
		t.Fatalf("resident bytes=%d want reachable weights=%d", f.runtime.StaticBytes(), expected)
	}
}
