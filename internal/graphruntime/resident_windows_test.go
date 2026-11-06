//go:build windows

package graphruntime

import (
	"slices"
	"testing"

	"overgo/internal/checked"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestResidentSessionReclaimsReleasedRecords holds a session that binds and
// releases each request to bookkeeping that does not grow with them: a shared
// binding keeps its record and its values through a partial release, the last
// release empties the list, a repeated release changes nothing, and a sealed
// session, which has no lookup map, reclaims its records the same way.
func TestResidentSessionReclaimsReleasedRecords(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	session := testcheck.Must(NewResidentSession(0))(t)
	defer func() {
		if err := session.Close(ctx); err != nil {
			t.Error(err)
		}
	}()
	builder := tensor.NewBuilder()
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(2))
	input := builder.Input("input", dtype.F32, tensor.MustShape(2))
	output := builder.Add(input, weight)
	testcheck.NoError(t, builder.Err())
	bind := func() *ResidentProgram {
		t.Helper()
		program := testcheck.Must(session.Compile(ctx, "records", nil, output))(t)
		testcheck.NoError(t, session.BindF32(ctx, program, "records", weight, []float32{1, 2}))
		return program
	}
	check := func(program *ResidentProgram) {
		t.Helper()
		result := testcheck.Must(session.Execute(ctx, program, map[*tensor.Tensor]reference.Value{input: {Shape: input.Shape, Data: []float32{10, 20}}}))(t)
		if want := []float32{11, 22}; !slices.Equal(result[output].Data, want) {
			t.Fatal("output", result[output].Data, "want", want)
		}
	}
	release := func(records int, programs ...*ResidentProgram) {
		t.Helper()
		testcheck.NoError(t, session.Release(ctx, programs...))
		if len(session.allocations) != records {
			t.Fatalf("%d records after release, want %d", len(session.allocations), records)
		}
	}
	for range 3 {
		program := bind()
		check(program)
		release(0, program)
		release(0, program)
		if session.StaticBytes() != 0 {
			t.Fatal("released weights remain", session.StaticBytes())
		}
	}
	first, second := bind(), bind()
	release(1, first)
	check(second)
	release(0, second)
	sealed := bind()
	session.Seal()
	check(sealed)
	release(0, sealed)
}

func TestCompiledFeedsUseIndexedInputs(t *testing.T) {
	builder := tensor.NewBuilder()
	dynamic := builder.Input("dynamic", dtype.F32, tensor.MustShape(2))
	static := builder.Input("static", dtype.F32, tensor.MustShape(2))
	output := builder.Add(dynamic, static)
	program := testcheck.Must(compileResidentProgram("resident test", []*tensor.Tensor{dynamic}, []*tensor.Tensor{output}))(t)
	dynamicSlot, dynamicOK := program.Graph.InputSlot(dynamic)
	staticSlot, staticOK := program.Graph.InputSlot(static)
	if !dynamicOK || !staticOK || len(program.dynamic) != 1 || program.dynamic[0] != dynamicSlot {
		t.Fatalf("compiled slots dynamic=%v static=%v program=%v", dynamicOK, staticOK, program.dynamic)
	}
	if checked.Nonzero(program.Inputs.Pointers[dynamicSlot]) || checked.Nonzero(program.Inputs.Pointers[staticSlot]) {
		t.Fatal("compiled input slots must start unbound")
	}
}
