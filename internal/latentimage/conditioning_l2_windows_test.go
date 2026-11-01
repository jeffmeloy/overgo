//go:build windows

package latentimage

import (
	"encoding/json"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"slices"
	"testing"
)

func TestMOSSConditioningL2Golden(t *testing.T) {
	cudatest.Require(t)
	const file = "l2-cuda.safetensors"
	mossFixtureBytes(t, file, "c57437804e63dd1de7feb69b6bbe1b0e2e0405c179a7b5e3480a8139cf59e229")
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", file))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	policies := map[string]float32{}
	for _, metadata := range s.Metadata {
		if err = json.Unmarshal([]byte(metadata["policies"]), &policies); err != nil {
			t.Fatal(err)
		}
	}
	if len(policies) != 10 {
		t.Fatal("native L2 coverage", len(policies))
	}
	run, _, close := cudaVAERunner(t)
	defer close()
	var names []string
	for name := range policies {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			b := tensor.NewBuilder()
			dims := slices.Clone(s.Tensors[name+".input"].Shape)
			slices.Reverse(dims)
			x := b.Input("input", dtype.F32, tensor.MustShape(dims...))
			out := b.L2NormNativeBF16(x, policies[name])
			if b.Err() != nil {
				t.Fatal(b.Err())
			}
			feeds := map[*tensor.Tensor]reference.Value{x: {Shape: x.Shape, Data: refinementValues(t, s, name+".input")}}
			want := refinementValues(t, s, name+".output")
			compare := func(label string, got []float32) {
				failed, unequal, worst := compareMOSSValues(got, want, mossStageTolerance{})
				t.Logf("%s values=%d failed=%d unequal=%d max_abs=%g", label, len(want), failed, unequal, worst)
				if failed != 0 || unequal != 0 {
					t.Error(label, "native differs")
				}
			}
			cpu, err := reference.Execute([]*tensor.Tensor{out}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			compare("cpu", cpu[out].Data)
			for pass := range 2 {
				got, err := run([]*tensor.Tensor{out}, feeds)
				if err != nil {
					t.Fatal(err)
				}
				compare([]string{"cuda", "repeat"}[pass], got[out].Data)
			}
		})
	}
}
