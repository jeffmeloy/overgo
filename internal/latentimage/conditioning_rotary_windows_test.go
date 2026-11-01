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

func TestMOSSConditioningRotaryGolden(t *testing.T) {
	cudatest.Require(t)
	const file = "rotary-cuda.safetensors"
	mossFixtureBytes(t, file, "6db58226427f7bc155b2856b011655042e10956843910b3ada87def731255845")
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", file))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	policies := map[string]struct {
		Rotary   uint32
		Base     float32
		Multi    bool
		Sections [4]int32
	}{}
	for _, metadata := range s.Metadata {
		if err = json.Unmarshal([]byte(metadata["policies"]), &policies); err != nil {
			t.Fatal(err)
		}
	}
	if len(policies) != 9 {
		t.Fatal("native rotary case coverage", len(policies))
	}
	run, _, close := cudaVAERunner(t)
	defer close()
	var names []string
	for name := range policies {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		for _, dynamic := range []bool{false, true} {
			t.Run(name+map[bool]string{false: "/static", true: "/dynamic"}[dynamic], func(t *testing.T) {
				policy := policies[name]
				b := tensor.NewBuilder()
				item := s.Tensors[name+".input"]
				dims := slices.Clone(item.Shape)
				slices.Reverse(dims)
				x := b.Input("input", dtype.F32, tensor.MustShape(dims...))
				feeds := map[*tensor.Tensor]reference.Value{x: {Shape: x.Shape, Data: refinementValues(t, s, name+".input")}}
				positions := refinementValues(t, s, name+".positions")
				tokens := x.Shape.Dims[2]
				var axes [4][]uint32
				for axis := range axes {
					axes[axis] = make([]uint32, tokens)
					for pos := range tokens {
						axes[axis][pos] = uint32(positions[pos*4+uint64(axis)])
					}
				}
				options := tensor.RoPEOptions{NativeBF16: true, Layout: tensor.RoPELayoutNeoX, RotaryDimensions: policy.Rotary, FrequencyBase: policy.Base, FrequencyScale: 1}
				if policy.Multi || dynamic {
					options.Sections = policy.Sections
					options.InterleavedSections = true
					if dynamic {
						p := b.Input("positions", dtype.F32, tensor.MustShape(4, 1, tokens))
						feeds[p] = reference.Value{Shape: p.Shape, Data: positions}
						options.PositionInput = p
					} else {
						options.MultiPositions = &axes
					}
				} else {
					options.Positions = axes[0]
				}
				out := b.RoPEWithOptions(x, options)
				if b.Err() != nil {
					t.Fatal(b.Err())
				}
				inverse := out.Inputs[len(out.Inputs)-1].Attrs.(tensor.EmbeddedInputAttributes).Data
				compare := func(label string, got, want []float32) {
					if len(got) != len(want) {
						t.Fatal(label, "length")
					}
					unequal := 0
					worst := 0.0
					for i, value := range got {
						if math.Float32bits(value) != math.Float32bits(want[i]) {
							unequal++
							worst = max(worst, math.Abs(float64(value)-float64(want[i])))
						}
					}
					t.Logf("%s values=%d unequal=%d max_abs=%g", label, len(want), unequal, worst)
					if unequal != 0 {
						t.Error(label, "native differs")
					}
				}
				compare("frequencies", inverse, refinementValues(t, s, name+".inverse"))
				want := refinementValues(t, s, name+".output")
				cpu, err := reference.Execute([]*tensor.Tensor{out}, feeds)
				if err != nil {
					t.Fatal(err)
				}
				compare("cpu", cpu[out].Data, want)
				got, err := run([]*tensor.Tensor{out}, feeds)
				if err != nil {
					t.Fatal(err)
				}
				compare("cuda", got[out].Data, want)
				again, err := run([]*tensor.Tensor{out}, feeds)
				if err != nil {
					t.Fatal(err)
				}
				compare("repeat", again[out].Data, got[out].Data)
			})
		}
	}
}
