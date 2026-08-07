package planner

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
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

func TestBuildAliasesReshapeStorageAndLifetime(t *testing.T) {
	const fixtureWidth = 4
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(fixtureWidth)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	computed := builder.Add(left, right)
	view := builder.Reshape(computed, 2, fixtureWidth/2)
	output := builder.Scale(view, 2)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	plan, err := Build([]*tensor.Tensor{output}, 16)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Allocations[computed].Offset != plan.Allocations[view].Offset {
		t.Fatalf("reshape storage differs: computed=%+v view=%+v", plan.Allocations[computed], plan.Allocations[view])
	}
	if plan.Allocations[computed].Last < plan.Allocations[output].First {
		t.Fatalf("reshape root expired before consumer: %+v", plan.Allocations[computed])
	}
}

func TestBuildWithDependenciesExtendsProducerLifetime(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	input := builder.Input("input", dtype.F32, shape)
	producer := builder.Scale(input, 2)
	intermediate := builder.SiLU(producer)
	overwriteCandidate := builder.Scale(input, 3)
	consumer := builder.Multiply(intermediate, overwriteCandidate)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	plan, err := BuildWithDependencies(
		[]*tensor.Tensor{consumer}, 16,
		map[*tensor.Tensor][]*tensor.Tensor{consumer: {producer}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Allocations[producer].Last < plan.Allocations[consumer].First {
		t.Fatalf("producer expired before rewritten consumer: %+v", plan.Allocations[producer])
	}
	if plan.Allocations[producer].Offset == plan.Allocations[overwriteCandidate].Offset {
		t.Fatal("extended producer shares storage with an intervening tensor")
	}
}

func TestBuildWithRewritesAliasesViewsAndOmitsFusedNodes(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8)
	input := builder.Input("input", dtype.F32, shape)
	producer := builder.Scale(input, 2)
	view := builder.FlatSlice(producer, 2, 4)
	eliminated := builder.SiLU(view)
	output := builder.Scale(eliminated, 3)
	plan, err := BuildWithRewrites(
		[]*tensor.Tensor{output},
		16,
		map[*tensor.Tensor][]*tensor.Tensor{output: {view}},
		map[*tensor.Tensor]*tensor.Tensor{view: producer},
		map[*tensor.Tensor]struct{}{eliminated: {}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, allocated := plan.Allocations[eliminated]; allocated {
		t.Fatal("eliminated tensor received arena storage")
	}
	if plan.Allocations[view].Offset != plan.Allocations[producer].Offset {
		t.Fatalf("view storage differs: producer=%+v view=%+v", plan.Allocations[producer], plan.Allocations[view])
	}
	if plan.Allocations[producer].Last < plan.Allocations[output].First {
		t.Fatalf("aliased producer expired before rewritten consumer: %+v", plan.Allocations[producer])
	}
}
