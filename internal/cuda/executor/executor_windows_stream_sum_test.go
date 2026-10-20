//go:build windows

package executor

import (
	"context"
	"testing"

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
		cuda := newFixtureExecutor(t)
		run := func(outputs ...*tensor.Tensor) []float32 {
			program, err := tensor.CompileProgram(outputs...)
			if err != nil {
				t.Fatal(err)
			}
			got, err := cuda.ExecuteProgram(context.WithoutCancel(t.Context()), program, feeds)
			if err != nil {
				t.Fatal(err)
			}
			return got[output].Data
		}
		compareBits(t, run(output), run(output, first))
		checkCUDAGraph(t, feeds, graphOutputCheck{output: output, tolerance: accuracyProjection})
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
