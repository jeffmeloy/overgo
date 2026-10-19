package latentimage

import (
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func conditioningDeltaGolden(t *testing.T, file, digest string, run graphruntime.Runner) {
	t.Helper()
	mossFixtureBytes(t, file, digest)
	s, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", file))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var cases []string
	for name := range s.Tensors {
		if before, ok := strings.CutSuffix(name, ".query"); ok {
			cases = append(cases, before)
		}
	}
	slices.Sort(cases)
	if len(cases) != 9 {
		t.Fatalf("native case coverage=%d", len(cases))
	}
	for _, name := range cases {
		for _, repeat := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/interleave=%v", name, repeat), func(t *testing.T) {
				read := func(key string) []float32 { return refinementValues(t, s, name+"."+key) }
				shape := s.Tensors[name+".value"].Shape
				batches, tokens, heads, width := int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
				keyHeads, qHeads := int(s.Tensors[name+".key"].Shape[2]), int(s.Tensors[name+".query"].Shape[2])
				sourceHead := func(h int) int {
					if repeat {
						return h
					}
					return (h%keyHeads)*(heads/keyHeads) + h/keyHeads
				}
				reorder := func(values []float32, columns int) []float32 {
					result := make([]float32, len(values))
					for batch := range batches {
						for token := range tokens {
							for head := range heads {
								src := ((batch*tokens+token)*heads + sourceHead(head)) * columns
								dst := ((batch*tokens+token)*heads + head) * columns
								copy(result[dst:dst+columns], values[src:src+columns])
							}
						}
					}
					return result
				}
				stateLayout := func(values []float32) []float32 {
					result := make([]float32, len(values))
					for batch := range batches {
						for head := range heads {
							for k := range width {
								for v := range width {
									result[((batch*heads+head)*width+v)*width+k] = values[((batch*heads+sourceHead(head))*width+k)*width+v]
								}
							}
						}
					}
					return result
				}
				shapes := []tensor.Shape{
					tensor.MustShape(uint64(width), uint64(qHeads), uint64(tokens), uint64(batches)),
					tensor.MustShape(uint64(width), uint64(keyHeads), uint64(tokens), uint64(batches)),
					tensor.MustShape(uint64(width), uint64(heads), uint64(tokens), uint64(batches)),
					tensor.MustShape(1, uint64(heads), uint64(tokens), uint64(batches)),
					tensor.MustShape(1, uint64(heads), uint64(tokens), uint64(batches)),
					tensor.MustShape(uint64(width), uint64(width), uint64(heads), uint64(batches)),
				}
				data := [][]float32{read("query"), read("key"), reorder(read("value"), width), reorder(read("g"), 1), reorder(read("beta"), 1), stateLayout(read("initial_state"))}
				b := tensor.NewBuilder()
				nodes := make([]*tensor.Tensor, len(data))
				feeds := map[*tensor.Tensor]reference.Value{}
				for i := range nodes {
					nodes[i] = b.Input(fmt.Sprint(i), dtype.F32, shapes[i])
					feeds[nodes[i]] = reference.Value{Shape: shapes[i], Data: data[i]}
				}
				out := b.GatedDeltaNetChunked(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], repeat)
				if err := b.Err(); err != nil {
					t.Fatal(err)
				}
				got, err := run([]*tensor.Tensor{out}, feeds)
				if err != nil {
					t.Fatal(err)
				}
				again, err := run([]*tensor.Tensor{out}, feeds)
				if err != nil {
					t.Fatal(err)
				}
				actual := got[out].Data
				for i, v := range actual {
					if math.Float32bits(v) != math.Float32bits(again[out].Data[i]) {
						t.Fatal("chunk update does not repeat exactly")
					}
				}
				wantOutput, wantState := reorder(read("output"), width), stateLayout(read("final_state"))
				if len(actual) != len(wantOutput)+len(wantState) {
					t.Fatal("packed output length differs")
				}
				for i, want := range append(wantOutput, wantState...) {
					value, relative := actual[i], 1.3e-6
					if i < len(wantOutput) {
						value = dtype.RoundBF16(value)
						relative = .016
					}
					difference := math.Abs(float64(value) - float64(want))
					if math.IsNaN(difference) || math.IsInf(difference, 0) || difference > 1e-5+relative*math.Abs(float64(want)) {
						t.Fatalf("packed value %d differs: got=%g native=%g", i, value, want)
					}
				}
			})
		}
	}
}

func TestMOSSConditioningDeltaCPU(t *testing.T) {
	conditioningDeltaGolden(t, "chunk-state-cpu.safetensors", "ffe6b10cba343db02610256fd92cbed2abce5df3bd597f0b7a73f751428b715a", reference.Execute)
}
