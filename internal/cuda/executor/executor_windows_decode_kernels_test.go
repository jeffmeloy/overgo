//go:build windows

package executor

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestDecodeKernelsMatchTheirReferences runs the decode kernels of Qwen3.8-Flash-Next's top products on its
// one-token shapes and holds each to the host reference:
//   - Q8_0 weights launch mul_mat_q8_0_input_f32 with the row's warps q8_0RowWarps picks, against the
//     dequantized weights times the input as the q8 input quantization rounds it
//   - F32 weights launch mul_mat_f32, a block a row, in place of cuBLAS
//   - a native routed layer launches moe_router_dot_f32 for its router logits
//
// Speculative spans of three columns and rows that miss 16-byte alignment take the same kernels.
func TestDecodeKernelsMatchTheirReferences(t *testing.T) {
	// Serial: device fixtures share the one card.
	// A launch record is four header words, then eight bytes a pointer and four a scalar.
	const headerBytes = 4 * 8
	word, scalar := binary.LittleEndian.Uint64, binary.LittleEndian.Uint32
	traced := func(t *testing.T, cuda *Executor) *device.LaunchTrace {
		trace := &device.LaunchTrace{}
		if err := cuda.worker.Do(t.Context(), func(state *device.State) error { state.Trace = trace; return nil }); err != nil {
			t.Fatal(err)
		}
		return trace
	}
	// run multiplies a weight stored as storage by columns input columns and returns the
	// device output, the reference and the launch trace.
	run := func(t *testing.T, storage dtype.Type, inner, rows, columns uint64) ([]float32, []float32, []byte) {
		name := fmt.Sprintf("decode-kernels/%s/%dx%dx%d", storage, inner, rows, columns)
		weights := moePinnedValues(name, 1, inner*rows, 3/math.Sqrt(float64(inner)))
		// Full-precision inputs: a BF16 value over its block's scale often lands on a half, which the
		// device's fast division rounds either way.
		draws := rand.New(rand.NewPCG(inner*rows, columns))
		input := make([]float32, inner*columns)
		for index := range input {
			input[index] = 2*draws.Float32() - 1
		}
		stored, used, referenceInput := driver.Bytes(weights), weights, input
		if storage == dtype.Q8_0 {
			var err error
			if stored, err = quant.Quantize(storage, weights); err != nil {
				t.Fatal(err)
			}
			if used, err = quant.Dequantize(storage, stored, inner*rows); err != nil {
				t.Fatal(err)
			}
			referenceInput = q8InputRoundTrip(input)
		}
		leftShape, rightShape := tensor.MustShape(inner, rows), tensor.MustShape(inner, columns)
		referenceBuilder := tensor.NewBuilder()
		referenceLeft := referenceBuilder.Input("left", dtype.F32, leftShape)
		referenceRight := referenceBuilder.Input("right", dtype.F32, rightShape)
		referenceOutput := referenceBuilder.MulMat(referenceLeft, referenceRight)
		want, err := reference.Execute([]*tensor.Tensor{referenceOutput}, map[*tensor.Tensor]reference.Value{
			referenceLeft: {Shape: leftShape, Data: used}, referenceRight: {Shape: rightShape, Data: referenceInput},
		})
		if err != nil {
			t.Fatal(err)
		}
		builder := tensor.NewBuilder()
		builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
		left := builder.Input("left", storage, leftShape)
		right := builder.Input("right", dtype.F32, rightShape)
		output := builder.MulMat(left, right)
		worker := newFixtureWorker(t)
		pointer := copyFixtureDeviceBytes(t, worker, stored)
		cuda := newFixtureExecutorWithWorker(t, worker)
		trace := traced(t, cuda)
		got, err := cuda.executeWithDeviceFeeds(context.WithoutCancel(t.Context()), []*tensor.Tensor{output},
			map[*tensor.Tensor]reference.Value{right: {Shape: rightShape, Data: input}},
			map[*tensor.Tensor]driver.DevicePtr{left: pointer})
		if err != nil {
			t.Fatal(err)
		}
		return got[output].Data, want[referenceOutput].Data, trace.Data()
	}

	// Flash-Next's Q8_0 products: linear-attention qkv, ssm_out, the hyper-connections' down and up,
	// the shared expert's gate and down; the 320 long rows share each row among q8_0SplitWarps warps.
	for _, shape := range []struct {
		inner, rows, columns uint64
		rowWarps             uint32
	}{
		{2560, 10240, 1, 1}, {6144, 2560, 1, 1}, {10240, 320, 1, q8_0SplitWarps}, {320, 10240, 1, 1},
		{2560, 640, 1, 1}, {640, 2560, 1, 1}, {10240, 320, 3, q8_0SplitWarps}, {640, 2560, 3, 1},
	} {
		t.Run(fmt.Sprintf("q8_0-%dx%dx%d", shape.inner, shape.rows, shape.columns), func(t *testing.T) {
			got, want, raw := run(t, dtype.Q8_0, shape.inner, shape.rows, shape.columns)
			// The input's quantization launches first: two pointers and a scalar.
			product := raw[headerBytes+2*8+4:]
			const rowWarpsOffset = headerBytes + 3*8 + 3*4
			if len(product) < rowWarpsOffset+4 || word(product) != uint64(kernelMulMatQ80InputF32) {
				t.Fatalf("second launch is not mul_mat_q8_0_input_f32 with its row warps")
			}
			if rowWarps := scalar(product[rowWarpsOffset:]); rowWarps != shape.rowWarps {
				t.Fatalf("row warps %d, want %d", rowWarps, shape.rowWarps)
			}
			compare(t, got, want, accuracyModel)
		})
	}

	// Flash-Next's F32 products: the hyper-connections' 4-row injections, the gated delta net's
	// 48-row alpha and beta, the shared expert's one-row gate; 2561 weights a row miss 16-byte alignment.
	for _, shape := range []struct{ inner, rows, columns uint64 }{
		{10240, 4, 1}, {2560, 48, 1}, {2560, 1, 1}, {2560, 48, 3}, {2561, 48, 1},
	} {
		t.Run(fmt.Sprintf("f32-%dx%dx%d", shape.inner, shape.rows, shape.columns), func(t *testing.T) {
			got, want, raw := run(t, dtype.F32, shape.inner, shape.rows, shape.columns)
			if len(raw) < headerBytes || word(raw) != uint64(kernelMulMatF32) {
				t.Fatal("first launch is not mul_mat_f32")
			}
			compare(t, got, want, accuracyModel)
		})
	}

	// Flash-Next's router: 2560 channels, 512 experts, ten routed, one decode token and a span of three.
	for _, tokens := range []uint64{1, 3} {
		t.Run(fmt.Sprintf("router-tokens=%d", tokens), func(t *testing.T) {
			c := moePinnedCase{
				name: "decode-kernels-router", compute: tensor.MulMatComputeNativeTensorCore,
				hidden: 2560, routerHidden: 2560, intermediate: 8, experts: 512, tokens: tokens, divisor: 1, topK: 10,
				routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1,
				reference: true,
			}
			cuda := newFixtureExecutor(t)
			trace := traced(t, cuda)
			got, want := c.run(t, cuda, dtype.F32)
			if raw := trace.Data(); len(raw) < headerBytes || word(raw) != uint64(kernelMoeRouterDotF32) {
				t.Fatal("first launch is not moe_router_dot_f32")
			}
			compare(t, got, want, accuracyModel)
		})
	}
}

// q8InputRoundTrip returns input as the q8 input quantization holds it: each 32-value block
// becomes its quants times its scale, its largest magnitude over 127.
func q8InputRoundTrip(input []float32) []float32 {
	const quantLimit = 127
	block := int(q8InputTraits.BlockSize)
	output := make([]float32, len(input))
	for start := 0; start < len(input); start += block {
		var maximum float32
		for _, value := range input[start : start+block] {
			maximum = max(maximum, float32(math.Abs(float64(value))))
		}
		scale := maximum / quantLimit
		for index, value := range input[start : start+block] {
			if maximum != 0 {
				quantized := math.RoundToEven(float64(value / scale))
				output[start+index] = float32(max(-quantLimit, min(quantLimit, quantized))) * scale
			}
		}
	}
	return output
}
