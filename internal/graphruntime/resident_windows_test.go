//go:build windows

package graphruntime

import (
	"testing"

	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestCompiledFeedsUseIndexedInputs(t *testing.T) {
	builder := tensor.NewBuilder()
	dynamic := builder.Input("dynamic", dtype.F32, tensor.MustShape(2))
	static := builder.Input("static", dtype.F32, tensor.MustShape(2))
	output := builder.Add(dynamic, static)
	program, err := compileResidentProgram("resident test", []*tensor.Tensor{dynamic}, []*tensor.Tensor{output})
	if err != nil {
		t.Fatal(err)
	}
	dynamicSlot, dynamicOK := program.Graph.InputSlot(dynamic)
	staticSlot, staticOK := program.Graph.InputSlot(static)
	if !dynamicOK || !staticOK || len(program.dynamic) != 1 || program.dynamic[0] != dynamicSlot {
		t.Fatalf("compiled slots dynamic=%v static=%v program=%v", dynamicOK, staticOK, program.dynamic)
	}
	if checked.Nonzero(program.Inputs.Pointers[dynamicSlot]) || checked.Nonzero(program.Inputs.Pointers[staticSlot]) {
		t.Fatal("compiled input slots must start unbound")
	}
}
