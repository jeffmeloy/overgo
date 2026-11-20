//go:build windows

package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestARoundedSITUFoldsIntoOneKernel holds MOSS's rounded SITU activation, built as the denoiser builds it, to one
// launch for the chain, and to changing no bit against the same chain launched operation by operation, which a graph
// that also keeps the gate's bound as an output does.
func TestARoundedSITUFoldsIntoOneKernel(t *testing.T) {
	// Serial: device fixtures share the one card.
	const gateBound, hiddenBound = 4, 25
	build := func(keep bool) (*CompiledGraph, *tensor.Tensor, map[*tensor.Tensor]reference.Value) {
		b := tensor.NewBuilder()
		gate := b.Input("gate", dtype.F32, tensor.MustShape(96, 24))
		up := b.Input("up", dtype.F32, tensor.MustShape(96, 24))
		round := b.BF16RoundNative
		bounded := func(x *tensor.Tensor, bound float32) *tensor.Tensor {
			return round(b.Scale(round(b.Tanh(round(b.Scale(x, tensor.UnitScale/bound)))), bound))
		}
		boundedGate := bounded(gate, gateBound)
		situ := round(b.Multiply(round(b.Multiply(boundedGate, round(b.Sigmoid(gate)))), bounded(up, hiddenBound)))
		outputs := []*tensor.Tensor{situ}
		if keep {
			outputs = append(outputs, boundedGate)
		}
		compiled, err := Compile(outputs...)
		if err != nil {
			t.Fatal(err)
		}
		return compiled, situ, map[*tensor.Tensor]reference.Value{
			gate: patternedValue(gate.Shape, 1, 0.83, 0.05), up: patternedValue(up.Shape, 2, 3.7, -0.4),
		}
	}
	cuda := newFixtureExecutor(t)
	run := func(keep bool) []float32 {
		compiled, situ, feeds := build(keep)
		folded := 0
		for _, frame := range compiled.nodes {
			if frame.fusion != nil && frame.fusion.kind == compiledFusionMOSSSitu {
				folded++
			}
		}
		if want := map[bool]int{false: 1, true: 0}[keep]; folded != want || !keep && len(compiled.launches) != 1 {
			t.Fatalf("keeping the gate's bound %t, %d chains fold in %d launches, want %d", keep, folded, len(compiled.launches), want)
		}
		got, err := cuda.ExecuteCompiled(t.Context(), compiled, feeds, nil)
		if err != nil {
			t.Fatal(err)
		}
		return got[situ].Data
	}
	compareBits(t, run(false), run(true))
}

// TestASITUReadsItsProductsHalvesWhereTheyLie holds the folded SITU activation over the two halves of one gate-and-up
// product, split as the denoiser splits it (latentimage rotaryGraph.part), to one launch that copies neither half, and
// to changing no bit against the same chain whose halves a graph also keeps as outputs, so each is copied apart.
func TestASITUReadsItsProductsHalvesWhereTheyLie(t *testing.T) {
	// Serial: device fixtures share the one card.
	const gateBound, hiddenBound, width, tokens = 4, 25, 96, 24
	build := func(keep bool) (*CompiledGraph, *tensor.Tensor, map[*tensor.Tensor]reference.Value) {
		b := tensor.NewBuilder()
		product := b.Input("product", dtype.F32, tensor.MustShape(2*width, tokens))
		half := func(offset uint64) *tensor.Tensor {
			return b.Reshape(b.GroupSlice(product, offset, width, tensor.SingletonExtent, width), width, tokens)
		}
		up, gate := half(0), half(width)
		round := b.BF16RoundNative
		bounded := func(x *tensor.Tensor, bound float32) *tensor.Tensor {
			return round(b.Scale(round(b.Tanh(round(b.Scale(x, tensor.UnitScale/bound)))), bound))
		}
		situ := round(b.Multiply(round(b.Multiply(bounded(gate, gateBound), round(b.Sigmoid(gate)))), bounded(up, hiddenBound)))
		outputs := []*tensor.Tensor{situ}
		if keep {
			outputs = append(outputs, gate, up)
		}
		return testcheck.Must(Compile(outputs...))(t), situ, map[*tensor.Tensor]reference.Value{product: patternedValue(product.Shape, 3, 2.9, -0.2)}
	}
	cuda := newFixtureExecutor(t)
	run := func(keep bool) []float32 {
		compiled, situ, feeds := build(keep)
		halves := 0
		for _, frame := range compiled.nodes {
			if frame.fusion != nil && frame.fusion.kind == compiledFusionMOSSSitu && frame.fusion.situ.halves != 0 {
				halves++
			}
		}
		if want := map[bool]int{false: 1, true: 0}[keep]; halves != want || !keep && len(compiled.launches) != 1 {
			t.Fatalf("keeping the halves %t, %d activations read the product's halves in %d launches, want %d", keep, halves, len(compiled.launches), want)
		}
		return testcheck.Must(cuda.ExecuteCompiled(t.Context(), compiled, feeds, nil))(t)[situ].Data
	}
	compareBits(t, run(false), run(true))
}
