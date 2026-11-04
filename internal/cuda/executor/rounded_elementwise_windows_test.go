//go:build windows

package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestABF16RoundingFoldsIntoTheProductItRounds holds a BF16 rounding of each elementwise product with a rounding kernel
// (a sum and a product, each also broadcast, a scale, a tanh, a sigmoid and an RMS norm by the native mean) to folding
// into the product's launch, which then launches nothing of its own, and to changing no bit: each rounded value equals
// the same rounding launched after its product, which a graph that also keeps the product as an output does.
func TestABF16RoundingFoldsIntoTheProductItRounds(t *testing.T) {
	// Serial: device fixtures share the one card.
	build := func(keep bool) (*CompiledGraph, []*tensor.Tensor, map[*tensor.Tensor]reference.Value) {
		b := tensor.NewBuilder()
		left := b.Input("left", dtype.F32, tensor.MustShape(64, 16))
		right := b.Input("right", dtype.F32, tensor.MustShape(64, 16))
		row := b.Input("row", dtype.F32, tensor.MustShape(64, 1))
		var rounded, products []*tensor.Tensor
		round := func(product *tensor.Tensor) *tensor.Tensor {
			products = append(products, product)
			rounded = append(rounded, b.BF16Round(product))
			return rounded[len(rounded)-1]
		}
		value := round(b.Add(left, right))
		value = round(b.Multiply(value, right))
		value = round(b.Add(value, row))
		value = round(b.Multiply(value, row))
		value = round(b.Scale(value, 0.7))
		value = round(b.Tanh(value))
		value = round(b.Sigmoid(value))
		round(b.RMSNormNativeMean(value, 1e-6))
		outputs := rounded
		if keep {
			outputs = append(append([]*tensor.Tensor(nil), rounded...), products...)
		}
		compiled, err := Compile(outputs...)
		if err != nil {
			t.Fatal(err)
		}
		feeds := map[*tensor.Tensor]reference.Value{
			left: patternedValue(left.Shape, 1, 0.37, 0.01), right: patternedValue(right.Shape, 2, 0.29, -0.02),
			row: patternedValue(row.Shape, 3, 0.11, 0.5),
		}
		return compiled, rounded, feeds
	}
	cuda := newFixtureExecutor(t)
	run := func(keep bool) []reference.Value {
		compiled, rounded, feeds := build(keep)
		folded := 0
		for _, frame := range compiled.nodes {
			if frame.fusion != nil && frame.fusion.kind == compiledFusionRoundedElementwise {
				folded++
			}
		}
		if want := map[bool]int{false: len(rounded), true: 0}[keep]; folded != want {
			t.Fatalf("keeping the products %t, %d of %d roundings fold into their products, want %d", keep, folded, len(rounded), want)
		}
		got, err := cuda.ExecuteCompiled(t.Context(), compiled, feeds, nil)
		if err != nil {
			t.Fatal(err)
		}
		values := make([]reference.Value, len(rounded))
		for index, node := range rounded {
			values[index] = got[node]
		}
		return values
	}
	folded, launched := run(false), run(true)
	for index := range folded {
		compareBits(t, folded[index].Data, launched[index].Data)
	}
}
