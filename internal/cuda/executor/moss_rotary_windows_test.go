//go:build windows

package executor

import (
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestARoundedRotaryFoldsIntoOneKernel holds MOSS's rounded rotary embedding, built as the denoiser builds it, to one
// launch a chain: alone and rounded again, and turning the queries and keys of a BF16 attention, whose own fusion reads
// past that second rounding. Each changes no bit against the same chains launched operation by operation, which a graph
// that also keeps the half-turned heads as outputs does.
func TestARoundedRotaryFoldsIntoOneKernel(t *testing.T) {
	// Serial: device fixtures share the one card.
	const width, heads, tokens = 64, 4, 12
	type graph struct {
		compiled *CompiledGraph
		outputs  []*tensor.Tensor
		feeds    map[*tensor.Tensor]reference.Value
	}
	build := func(keep bool) graph {
		b := tensor.NewBuilder()
		shape := tensor.MustShape(width, heads, tokens)
		query, key, value := b.Input("query", dtype.F32, shape), b.Input("key", dtype.F32, shape), b.Input("value", dtype.F32, shape)
		cosine := b.Input("cosine", dtype.F32, tensor.MustShape(width, 1, tokens))
		sine := b.Input("sine", dtype.F32, tensor.MustShape(width, 1, tokens))
		round := b.BF16RoundNative
		var turns []*tensor.Tensor
		rotate := func(x *tensor.Tensor) *tensor.Tensor {
			part := func(start uint64) *tensor.Tensor {
				return b.Reshape(b.GroupSlice(x, start, width/2, tensor.SingletonExtent, width/2), width/2, heads, tokens)
			}
			turned := b.Concat(b.Scale(part(width/2), -tensor.UnitScale), part(0), tensor.FirstOffset)
			turns = append(turns, turned)
			return round(round(b.Add(round(b.Multiply(x, cosine)), round(b.Multiply(turned, sine)))))
		}
		alone := rotate(value)
		scale := float32(float64(tensor.UnitScale) / math.Sqrt(width))
		attended := round(b.AttentionWithOptions(rotate(query), rotate(key), round(value), tensor.AttentionOptions{Scale: scale}))
		outputs := []*tensor.Tensor{alone, attended}
		compiledOutputs := outputs
		if keep {
			compiledOutputs = append(append([]*tensor.Tensor(nil), outputs...), turns...)
		}
		compiled, err := Compile(compiledOutputs...)
		if err != nil {
			t.Fatal(err)
		}
		feeds := map[*tensor.Tensor]reference.Value{
			query: patternedValue(shape, 1, 0.71, 0.03), key: patternedValue(shape, 4, 0.53, -0.02),
			value: patternedValue(shape, 5, 0.37, 0.01), cosine: patternedValue(cosine.Shape, 2, 0.11, 0.02),
			sine: patternedValue(sine.Shape, 3, 0.09, -0.01),
		}
		return graph{compiled, outputs, feeds}
	}
	cuda := newFixtureExecutor(t)
	run := func(keep bool) [][]float32 {
		g := build(keep)
		folded := 0
		for _, frame := range g.compiled.nodes {
			if frame.fusion != nil && frame.fusion.kind == compiledFusionMOSSRotary {
				folded++
			}
		}
		if want := map[bool]int{false: 3, true: 0}[keep]; folded != want {
			t.Fatalf("keeping the turned heads %t, %d chains fold, want %d", keep, folded, want)
		}
		got, err := cuda.ExecuteCompiled(t.Context(), g.compiled, g.feeds, nil)
		if err != nil {
			t.Fatal(err)
		}
		values := make([][]float32, len(g.outputs))
		for index, output := range g.outputs {
			values[index] = got[output].Data
		}
		return values
	}
	folded, launched := run(false), run(true)
	for index := range folded {
		compareBits(t, folded[index], launched[index])
	}
}
