package optimizer

import (
	"math"
	"testing"

	"overgo/internal/binaryschema"
)

func TestHostStateBytesMatchesAllocations(t *testing.T) {
	for _, frozen := range []bool{false, true} {
		plan, err := CompilePlan(30, []GroupSpec{
			{Name: "tall", End: 6, Rows: 3, Cols: 2},
			{Name: "wide", Start: 6, End: 14, Rows: 2, Cols: 4},
			{Name: "optional", Start: 14, End: 30, Rows: 4, Cols: 4, Frozen: frozen},
		})
		if err != nil {
			t.Fatal(err)
		}
		admitted, err := plan.HostStateBytes()
		if err != nil {
			t.Fatal(err)
		}
		weights, gradients := make([]float32, plan.ParameterCount()), make([]float32, plan.ParameterCount())
		for index := range gradients {
			gradients[index] = float32(index + 1)
		}
		state, err := New(weights, gradients, plan, fixtureConfig)
		if err != nil {
			t.Fatal(err)
		}
		state.Step()
		actual := uint64(cap(state.momentum)+cap(state.scratch.input)+cap(state.scratch.output)+cap(state.scratch.gram)+cap(state.scratch.square)) * binaryschema.Uint64Bytes
		if admitted != actual {
			t.Fatalf("frozen=%v admitted=%d, actual numeric storage=%d", frozen, admitted, actual)
		}
	}
	if _, err := (Plan{}).HostStateBytes(); err == nil {
		t.Fatal("uncompiled plan admitted")
	}
	empty, err := CompilePlan(0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if bytes, err := empty.HostStateBytes(); err != nil || bytes != 0 {
		t.Fatalf("empty plan storage = %d, %v", bytes, err)
	}
	huge, err := CompilePlan(math.MaxInt, []GroupSpec{{Name: "huge", End: math.MaxInt, Rows: 1, Cols: math.MaxInt}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := huge.HostStateBytes(); err == nil {
		t.Fatal("unaddressable optimizer state admitted")
	}
}
