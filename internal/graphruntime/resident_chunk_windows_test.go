//go:build windows

package graphruntime

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestAResidentProgramHoldsItsWeightsInOneAllocation binds three weights of sizes no page holds evenly into one
// program: the driver's ledger counts one allocation for all three, the program computes with each, and releasing it
// returns the card to what it held.
func TestAResidentProgramHoldsItsWeightsInOneAllocation(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	session, err := NewResidentSession(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := session.Close(ctx); err != nil {
			t.Error(err)
		}
	}()
	builder := tensor.NewBuilder()
	// Seven hundred, three hundred and forty kilobytes, each added to an input of its own size.
	sizes := []uint64{179200, 76800, 10240}
	var inputs, weights, outputs []*tensor.Tensor
	for index, elements := range sizes {
		name := string(rune('a' + index))
		inputs = append(inputs, builder.Input("input"+name, dtype.F32, tensor.MustShape(elements)))
		weights = append(weights, builder.Input("weight"+name, dtype.F32, tensor.MustShape(elements)))
		outputs = append(outputs, builder.Add(inputs[index], weights[index]))
	}
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	program, err := session.Compile(ctx, "chunk", nil, outputs...)
	if err != nil {
		t.Fatal(err)
	}
	before, err := session.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	host := map[*tensor.Tensor]reference.Value{}
	for index, weight := range weights {
		values, ones := make([]float32, sizes[index]), make([]float32, sizes[index])
		for at := range values {
			values[at], ones[at] = float32(index+1), 1
		}
		if err := session.BindF32(ctx, program, "chunk", weight, values); err != nil {
			t.Fatal(err)
		}
		host[inputs[index]] = reference.Value{Shape: inputs[index].Shape, Data: ones}
	}
	bound, err := session.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if made := bound.Device.Allocations - before.Device.Allocations; made != 1 {
		t.Fatalf("binding three weights made %d allocations, want one", made)
	}
	result, err := session.Execute(ctx, program, host)
	if err != nil {
		t.Fatal(err)
	}
	for index, out := range outputs {
		got := result[out].Data
		if want := float32(index + 2); len(got) != int(sizes[index]) || got[0] != want || got[len(got)-1] != want {
			t.Fatalf("output %d reads %v at its ends, want %v: a weight was not where its program reads it", index, []float32{got[0], got[len(got)-1]}, want)
		}
	}
	// The run's own execution memory goes first, so what release returns is the weights' alone.
	if err := session.ReleaseStageMemory(ctx); err != nil {
		t.Fatal(err)
	}
	if err := session.Release(ctx, program); err != nil {
		t.Fatal(err)
	}
	released, err := session.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if released.Device.CurrentBytes > before.Device.CurrentBytes || session.StaticBytes() != 0 {
		t.Fatalf("after release the session holds %d bytes (%d before) and %d static", released.Device.CurrentBytes, before.Device.CurrentBytes, session.StaticBytes())
	}
}
