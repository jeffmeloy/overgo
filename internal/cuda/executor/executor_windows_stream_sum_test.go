//go:build windows

package executor

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestExecutorStreamSumMatchesTheUnfusedGraph runs a mix's reduction of 4
// streams of 64 values over 3 tokens, gated and plain, as one stream-sum
// kernel and, with an add kept as an output, as its group slices, adds and
// scale: the sums agree bit for bit. One token's sum read by a Q8_0
// product, which takes its blocks from the stream-sum kernel, holds to the
// dequantized reference.
func TestExecutorStreamSumMatchesTheUnfusedGraph(t *testing.T) {
	cudatest.Require(t)
	const width, streams, tokens = 64, 4, 3
	for _, gated := range []bool{true, false} {
		builder := tensor.NewBuilder()
		gate := builder.Input("gate", dtype.F32, tensor.MustShape(width*streams, tokens))
		up := builder.Input("up", dtype.F32, gate.Shape)
		x := up
		if gated {
			x = builder.Multiply(builder.SiLU(gate), up)
		}
		output, first := streamMix(builder, x, width, streams)
		if compiled, err := Compile(output); err != nil || compiled.nodes[compiled.orderIndexes[output]].fusion == nil {
			t.Fatalf("gated %t: the reduction did not fuse: %v", gated, err)
		}
		feeds := map[*tensor.Tensor]reference.Value{
			gate: patternedValue(gate.Shape, 7, 0.37, -0.2),
			up:   patternedValue(up.Shape, 11, 0.21, 0.05),
		}
		matchesUnfused(t, feeds, output, first)
	}
	projectionShape := tensor.MustShape(width, 5)
	projection := patternedValue(projectionShape, 13, 0.03, -0.1)
	storage, err := quant.Quantize(dtype.Q8_0, projection.Data)
	if err != nil {
		t.Fatal(err)
	}
	dequantized, err := quant.Dequantize(dtype.Q8_0, storage, uint64(len(projection.Data)))
	if err != nil {
		t.Fatal(err)
	}
	const inputQuantizedDecodeTolerance = 8e-2
	checkResidentBinaryGraph(t, dtype.Q8_0, projectionShape, storage, dequantized,
		patternedValue(tensor.MustShape(width*streams, 1), 7, 0.05, 0.02),
		func(builder *tensor.Builder, left, right *tensor.Tensor) *tensor.Tensor {
			sum, _ := streamMix(builder, right, width, streams)
			return builder.MulMat(left, sum)
		}, inputQuantizedDecodeTolerance)
}

// TestExecutorQ8EmissionKeepsASpansColumns runs a hyper-connection mix's normalization and gate, each read by a Q8_0
// product that takes its blocks from the fused kernel, over a verify span's two columns and over each column alone:
// the span's fused kernels write each column's blocks where one column's do, so its products agree bit for bit.
func TestExecutorQ8EmissionKeepsASpansColumns(t *testing.T) {
	cudatest.Require(t)
	const width, streams, rank, span = 64, 4, 32, 2
	quantized := func(shape tensor.Shape, seed int) []byte {
		storage, err := quant.Quantize(dtype.Q8_0, patternedValue(shape, seed, 0.03, -0.1).Data)
		if err != nil {
			t.Fatal(err)
		}
		return storage
	}
	downShape, upShape := tensor.MustShape(width*streams, rank), tensor.MustShape(rank, width)
	downBytes, upBytes := quantized(downShape, 13), quantized(upShape, 17)
	inputs := patternedValue(tensor.MustShape(width, streams, span), 5, 0.31, -0.2)
	weights := patternedValue(tensor.MustShape(width, streams), 9, 0.07, 1)
	worker := newFixtureWorker(t)
	down, up := copyFixtureDeviceBytes(t, worker, downBytes), copyFixtureDeviceBytes(t, worker, upBytes)
	cuda := newFixtureExecutorWithWorker(t, worker)
	// run multiplies tokens columns of the inputs from first and returns both products.
	run := func(first, tokens uint64) (low, high []float32) {
		t.Helper()
		builder := tensor.NewBuilder()
		input := builder.Input("input", dtype.F32, tensor.MustShape(width, streams, tokens))
		weight := builder.Input("weight", dtype.F32, weights.Shape)
		downNode, upNode := builder.Input("down", dtype.Q8_0, downShape), builder.Input("up", dtype.Q8_0, upShape)
		normalized, _, lowNode, highNode := hyperNorms(builder, input, weight, downNode, upNode)
		compiled, err := Compile(highNode)
		if err != nil {
			t.Fatal(err)
		}
		if descriptor := compiled.nodes[compiled.orderIndexes[normalized]].fusion; descriptor == nil || !descriptor.emitQ8 {
			t.Fatalf("%d columns: the normalization writes no Q8_0 blocks", tokens)
		}
		columns := inputs.Data[first*width*streams : (first+tokens)*width*streams]
		got, err := cuda.executeWithDeviceFeeds(t.Context(), []*tensor.Tensor{lowNode, highNode},
			map[*tensor.Tensor]reference.Value{input: {Shape: input.Shape, Data: columns}, weight: weights},
			map[*tensor.Tensor]driver.DevicePtr{downNode: down, upNode: up})
		if err != nil {
			t.Fatal(err)
		}
		return got[lowNode].Data, got[highNode].Data
	}
	spanLow, spanHigh := run(0, span)
	for column := range uint64(span) {
		low, high := run(column, 1)
		if !slices.Equal(spanLow[column*rank:(column+1)*rank], low) || !slices.Equal(spanHigh[column*width:(column+1)*width], high) {
			t.Fatalf("column %d of the span differs from the column alone", column)
		}
	}
}

// TestExecutorStreamCombineMatchesTheUnfusedGraph runs a combine of 4
// streams of 64 values over 3 tokens as one stream-combine kernel, its
// scaled-sigmoid gate folded in, and, with its product kept as an output, as
// its repeat, product, add and the gate's three kernels: the outputs agree
// bit for bit.
func TestExecutorStreamCombineMatchesTheUnfusedGraph(t *testing.T) {
	cudatest.Require(t)
	const width, streams, tokens = 64, 4, 3
	builder := tensor.NewBuilder()
	residual := builder.Input("residual", dtype.F32, tensor.MustShape(width, streams, tokens))
	current := builder.Input("current", dtype.F32, tensor.MustShape(width, tokens))
	x := builder.Input("x", dtype.F32, tensor.MustShape(streams, tokens))
	output, product, _ := streamCombine(builder, residual, current, x)
	if compiled, err := Compile(output); err != nil || compiled.nodes[compiled.orderIndexes[output]].fusion == nil {
		t.Fatalf("the combine did not fuse: %v", err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		residual: patternedValue(residual.Shape, 5, 0.29, -0.4),
		current:  patternedValue(current.Shape, 9, 0.17, 0.1),
		x:        patternedValue(x.Shape, 13, 0.73, -0.3),
	}
	matchesUnfused(t, feeds, output, product)
}

// TestExecutorHyperNormsMatchTheUnfusedGraph runs a normalization of 4
// streams of 64 values over 3 tokens by one weight row a stream, and a SiLU
// of a scaled tensor, each as one fused kernel and, with an intermediate
// kept as an output, as the kernels it replaces: the outputs agree bit for
// bit. A one-token scaled SiLU read by a Q8_0 product, which takes its blocks
// from the fused kernel, holds to the dequantized reference.
func TestExecutorHyperNormsMatchTheUnfusedGraph(t *testing.T) {
	cudatest.Require(t)
	const width, streams, tokens = 64, 4, 3
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(width, streams, tokens))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(width, streams))
	normalization := builder.RMSNorm(input, 1e-6)
	normalized := builder.Multiply(normalization, weight)
	x := builder.Input("x", dtype.F32, tensor.MustShape(width, tokens))
	scaled := builder.Scale(x, 0.25)
	gate := builder.SiLU(scaled)
	feeds := map[*tensor.Tensor]reference.Value{
		input:  patternedValue(input.Shape, 5, 0.31, -0.2),
		weight: patternedValue(weight.Shape, 9, 0.07, 1),
		x:      patternedValue(x.Shape, 13, 0.53, 0.1),
	}
	for _, fused := range []struct{ output, kept *tensor.Tensor }{{normalized, normalization}, {gate, scaled}} {
		if compiled, err := Compile(fused.output); err != nil || compiled.nodes[compiled.orderIndexes[fused.output]].fusion == nil {
			t.Fatalf("tensor %d did not fuse: %v", fused.output.ID, err)
		}
		matchesUnfused(t, feeds, fused.output, fused.kept)
	}
	projectionShape := tensor.MustShape(width, 5)
	projection := patternedValue(projectionShape, 13, 0.03, -0.1)
	storage, err := quant.Quantize(dtype.Q8_0, projection.Data)
	if err != nil {
		t.Fatal(err)
	}
	dequantized, err := quant.Dequantize(dtype.Q8_0, storage, uint64(len(projection.Data)))
	if err != nil {
		t.Fatal(err)
	}
	const inputQuantizedDecodeTolerance = 8e-2
	checkResidentBinaryGraph(t, dtype.Q8_0, projectionShape, storage, dequantized,
		patternedValue(tensor.MustShape(width, 1), 7, 0.5, 0.02),
		func(builder *tensor.Builder, left, right *tensor.Tensor) *tensor.Tensor {
			return builder.MulMat(left, builder.SiLU(builder.Scale(right, 0.25)))
		}, inputQuantizedDecodeTolerance)
}

// matchesUnfused holds output, fused, to the reference, and its card result
// to the same graph's with kept an output too, which leaves it unfused, bit
// for bit.
func matchesUnfused(t *testing.T, feeds map[*tensor.Tensor]reference.Value, output, kept *tensor.Tensor) {
	t.Helper()
	fused := checkCUDAGraph(t, feeds, graphOutputCheck{output: output, tolerance: accuracyProjection})
	unfused, err := newFixtureExecutor(t).Execute(context.WithoutCancel(t.Context()), []*tensor.Tensor{output, kept}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compareBits(t, fused[output].Data, unfused[output].Data)
}
