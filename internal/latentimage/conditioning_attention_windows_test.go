//go:build windows

package latentimage

import (
	"encoding/json"
	"math"
	"path/filepath"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestMOSSConditioningAttentionGolden(t *testing.T) {
	testConditioningAttentionGolden(t, "attention-cuda.safetensors", "199ea7dd97daa70a80952b3b377d14f4d721f75ef975f959bf29e7d68b7306db", 12)
}

func TestMOSSConditioningLongAttentionGolden(t *testing.T) {
	testConditioningAttentionGolden(t, "attention-long-cuda.safetensors", "9afc4a5eeb81f1557cb8a9153c07a90ba0061d0bb02bff9cb406c02c5c60bb1c", 5)
}

func testConditioningAttentionGolden(t *testing.T, file, digest string, count int) {
	cudatest.Require(t)
	mossFixtureBytes(t, file, digest)
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", file))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	policies := map[string]struct {
		Scale  float32
		Causal bool
	}{}
	for _, metadata := range s.Metadata {
		if err = json.Unmarshal([]byte(metadata["policies"]), &policies); err != nil {
			t.Fatal(err)
		}
	}
	if len(policies) != count {
		t.Fatal("native case coverage", len(policies))
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
			feeds := map[*tensor.Tensor]reference.Value{}
			input := func(label string) *tensor.Tensor {
				item := s.Tensors[name+"."+label]
				dims := slices.Clone(item.Shape)
				slices.Reverse(dims)
				x := b.Input(label, dtype.F32, tensor.MustShape(dims...))
				feeds[x] = reference.Value{Shape: x.Shape, Data: refinementValues(t, s, name+"."+label)}
				return x
			}
			q, k, v := input("query"), input("key"), input("value")
			policy := policies[name]
			out := b.BF16Round(b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: policy.Scale, Causal: policy.Causal, NativeMath: true}))
			if b.Err() != nil {
				t.Fatal(b.Err())
			}
			got, err := run([]*tensor.Tensor{out}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			again, err := run([]*tensor.Tensor{out}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			want := refinementValues(t, s, name+".output")
			if len(got[out].Data) != len(want) {
				t.Fatal("output length")
			}
			unequal := 0
			worst := 0.0
			for i, value := range got[out].Data {
				if math.Float32bits(value) != math.Float32bits(want[i]) {
					unequal++
					worst = max(worst, math.Abs(float64(value)-float64(want[i])))
				}
				if math.Float32bits(value) != math.Float32bits(again[out].Data[i]) {
					t.Fatal("repeat differs")
				}
			}
			t.Logf("values=%d unequal=%d max_abs=%g", len(want), unequal, worst)
			if unequal != 0 {
				t.Fatal("native attention differs")
			}
		})
	}
}
