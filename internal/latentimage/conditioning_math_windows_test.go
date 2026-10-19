//go:build windows

package latentimage

import (
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestMOSSConditioningMathGolden(t *testing.T) {
	cudatest.Require(t)
	t.Run("portable", func(t *testing.T) {
		conditioningMathGolden(t, "math-cuda.safetensors", "faab752f7fc402a579c2f2dd7b80b532061eb75ceb464bbb7f3c6a417a3b0fee", 13)
	})
	t.Run("cross-warp", func(t *testing.T) {
		conditioningMathGolden(t, "math-reduction-cuda.safetensors", "e8ea2c998d9c05d0de09cb9e0936172406955855b9e2e851810b50be996bd51f", 3)
	})
}

func conditioningMathGolden(t *testing.T, file, digest string, cases int) {
	t.Helper()
	mossFixtureBytes(t, file, digest)
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", file))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	run, _, close := cudaVAERunner(t)
	defer close()
	var names []string
	for name := range s.Tensors {
		if before, ok := strings.CutSuffix(name, ".input"); ok {
			names = append(names, before)
		}
	}
	slices.Sort(names)
	if len(names) != cases {
		t.Fatal("native case coverage", len(names))
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			b := tensor.NewBuilder()
			feeds := map[*tensor.Tensor]reference.Value{}
			input := func(label string) *tensor.Tensor {
				item := s.Tensors[name+"."+label]
				dims := slices.Clone(item.Shape)
				slices.Reverse(dims)
				x := b.Input(label, dtype.F32, tensor.MustShape(dims...))
				feeds[x] = reference.Value{Shape: x.Shape, Data: refinementValues(t, s, name+"."+label)}
				return x
			}
			x := input("input")
			outputs := map[string]*tensor.Tensor{}
			if name == "scalar" {
				outputs["exp"] = b.ExpNative(x)
				outputs["softplus"] = b.SoftplusNative(x)
			} else {
				w, z := input("weight"), input("gate")
				one := b.Input("one", dtype.F32, tensor.MustShape(1))
				feeds[one] = reference.Value{Shape: one.Shape, Data: []float32{1}}
				norm := b.RMSNormNative(x, 1e-6)
				outputs["normalized"] = norm
				outputs["weighted"] = b.BF16Round(b.Multiply(norm, b.Add(one, w)))
				outputs["gated"] = b.BF16Round(b.Multiply(b.BF16Round(b.Multiply(b.BF16Round(norm), w)), b.SiLU(z)))
			}
			if b.Err() != nil {
				t.Fatal(b.Err())
			}
			var nodes []*tensor.Tensor
			for _, n := range outputs {
				nodes = append(nodes, n)
			}
			got, err := run(nodes, feeds)
			if err != nil {
				t.Fatal(err)
			}
			again, err := run(nodes, feeds)
			if err != nil {
				t.Fatal(err)
			}
			for label, node := range outputs {
				want := refinementValues(t, s, name+"."+label)
				actual := got[node].Data
				if len(actual) != len(want) {
					t.Fatal("output length")
				}
				for i, v := range actual {
					if math.Float32bits(v) != math.Float32bits(want[i]) {
						t.Errorf("%s[%d]=%g native=%g", label, i, v, want[i])
						break
					}
					if math.Float32bits(v) != math.Float32bits(again[node].Data[i]) {
						t.Fatal("repeat differs")
					}
				}
			}
		})
	}
}
