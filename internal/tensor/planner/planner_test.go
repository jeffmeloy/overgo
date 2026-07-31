package planner

import (
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

func TestBuildReusesExpiredAllocation(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	add := builder.Add(left, right)
	scale := builder.Scale(add, 2)
	output := builder.SiLU(scale)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	plan, err := Build([]*tensor.Tensor{output}, 16)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ArenaSize != 32 {
		t.Fatalf("arena size = %d, want 32", plan.ArenaSize)
	}
	if plan.Allocations[add].Offset != plan.Allocations[output].Offset {
		t.Fatalf(
			"expired allocation was not reused: add=%d output=%d",
			plan.Allocations[add].Offset,
			plan.Allocations[output].Offset,
		)
	}
}

func TestBuildKeepsOutputLive(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	input := builder.Input("input", dtype.F32, shape)
	first := builder.Scale(input, 2)
	second := builder.Scale(first, 3)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	plan, err := Build([]*tensor.Tensor{first, second}, 16)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Allocations[first].Offset == plan.Allocations[second].Offset {
		t.Fatal("two live outputs share an allocation")
	}
}
