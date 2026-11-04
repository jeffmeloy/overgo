//go:build windows

package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
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
