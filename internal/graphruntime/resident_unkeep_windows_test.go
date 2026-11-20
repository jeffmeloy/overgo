//go:build windows

package graphruntime

import (
	"slices"
	"testing"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestAnUnkeptSourceFreesWhatNoProgramBinds holds a session's own hold on the inputs it keeps (Keep) to their source
// (Unkeep, moss-merge-compute): unkeeping a source frees each input kept of it that no program binds, keeps one a
// program binds until the program is released, and leaves another source's kept, read where it lies, packed in the
// chunk the first kept input opened; and an input kept once all are freed is read anew into memory the card holds,
// not that freed chunk.
func TestAnUnkeptSourceFreesWhatNoProgramBinds(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	session := testcheck.Must(NewResidentSession(0))(t)
	defer func() { testcheck.NoError(t, session.Close(ctx)) }()
	// wide is charged a mid page (residentCharge), whose rest the next kept input packs in.
	const wideValues = residentPacked/4 + residentAlignment
	builder := tensor.NewBuilder()
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(2))
	input := builder.Input("input", dtype.F32, tensor.MustShape(2))
	wide := builder.Input("wide", dtype.F32, tensor.MustShape(wideValues))
	alone := builder.Input("alone", dtype.F32, tensor.MustShape(2))
	output := builder.Add(input, weight)
	testcheck.NoError(t, builder.Err())
	program := testcheck.Must(session.Compile(ctx, "unkeep", nil, output))(t)
	testcheck.NoError(t, session.BindF32(ctx, program, "old", weight, []float32{1, 2}))
	reads := 0
	keep := func(source string, node *tensor.Tensor, values ...float32) driver.DevicePtr {
		t.Helper()
		return testcheck.Must(session.Keep(ctx, source, node.Name, node, func() ([]byte, error) {
			reads++
			return driver.Bytes(values), nil
		}))(t)
	}
	lies := func(pointer driver.DevicePtr, want ...float32) {
		t.Helper()
		got := make([]float32, len(want))
		testcheck.NoError(t, session.worker.Do(ctx, func(state *device.State) error {
			return state.Driver.MemcpyDtoH(unsafe.Slice((*byte)(unsafe.Pointer(&got[0])), 4*len(got)), pointer)
		}))
		if !slices.Equal(got, want) {
			t.Errorf("a kept input reads %v; want %v", got, want)
		}
	}
	const pairBytes, wideBytes = 8, 4 * wideValues
	keep("old", weight, 1, 2)
	keep("old", wide, make([]float32, wideValues)...)
	other := keep("new", alone, 5, 6)
	if reads != 2 || session.StaticBytes() != 2*pairBytes+wideBytes || session.allocations[2].chunk != session.allocations[1].chunk {
		t.Fatalf("keeping a bound input and two others read %d payloads into %d bytes; want the two others alone, packed together, %d bytes in all",
			reads, session.StaticBytes(), 2*pairBytes+wideBytes)
	}
	testcheck.NoError(t, session.Unkeep(ctx, "old"))
	if session.StaticBytes() != 2*pairBytes {
		t.Errorf("unkeeping a source left %d bytes; want its input no program binds freed, %d", session.StaticBytes(), 2*pairBytes)
	}
	result := testcheck.Must(session.Execute(ctx, program, map[*tensor.Tensor]reference.Value{input: {Shape: input.Shape, Data: []float32{10, 20}}}))(t)
	if want := []float32{11, 22}; !slices.Equal(result[output].Data, want) {
		t.Errorf("a program binding an unkept input reads %v; want %v", result[output].Data, want)
	}
	lies(other, 5, 6)
	testcheck.NoError(t, session.Release(ctx, program))
	testcheck.NoError(t, session.Unkeep(ctx, "new"))
	if session.StaticBytes() != 0 || len(session.allocations) != 0 {
		t.Errorf("a session holding nothing keeps %d bytes in %d records", session.StaticBytes(), len(session.allocations))
	}
	lies(keep("old", alone, 7, 8), 7, 8)
	if reads != 3 {
		t.Errorf("an input kept again after its source was unkept read %d payloads in all; want 3", reads)
	}
}
