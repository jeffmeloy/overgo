//go:build windows

package executor

import (
	"context"
	"encoding/binary"
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestAnInt8ProductComputesItsNormedOrActivatedInput holds the int8 ConvRot products a MOSS block makes of a norm with
// its steps (two products of one, as q and k read one modulated norm) and of a SITU activation over one product's
// halves to quantizing that input as they compute it (applyInt8PrologueRewrite), never storing it, and to changing no
// bit against the same products of the input stored, which a graph that also keeps the input as an output does.
func TestAnInt8ProductComputesItsNormedOrActivatedInput(t *testing.T) {
	// Serial: device fixtures share the one card.
	const inner, out, tokens, epsilon, gateBound, hiddenBound = 256, 64, 24, 1e-6, 4, 25
	storage := func(seed int) []byte {
		bytes := make([]byte, out*inner+out*4)
		for index := range out * inner {
			bytes[index] = byte(int8((index*61+17*seed)%255 - 127))
		}
		for row := range out {
			binary.LittleEndian.PutUint32(bytes[out*inner+row*4:], math.Float32bits(float32(1+(row+seed)%7)/1024))
		}
		return bytes
	}
	worker := newFixtureWorker(t)
	weights := [3]driver.DevicePtr{}
	for index := range weights {
		weights[index] = copyFixtureDeviceBytes(t, worker, storage(index+1))
	}
	cuda := newFixtureExecutorWithWorker(t, worker)
	census := map[uint64]int{}
	testcheck.NoError(t, cuda.worker.Do(t.Context(), func(state *device.State) error {
		state.Trace = &device.LaunchTrace{Census: census}
		return nil
	}))
	t.Cleanup(func() {
		_ = cuda.worker.Do(context.WithoutCancel(t.Context()), func(state *device.State) error {
			state.Trace = nil
			return nil
		})
	})
	run := func(keep bool) ([][]float32, []string) {
		b := tensor.NewBuilder()
		round := b.BF16RoundNative
		stream := b.Input("stream", dtype.F32, tensor.MustShape(inner, tokens))
		weight := b.Input("weight", dtype.F32, tensor.MustShape(inner))
		scale := b.Input("scale", dtype.F32, tensor.MustShape(inner))
		shift := b.Input("shift", dtype.F32, tensor.MustShape(inner))
		fused := b.Input("fused", dtype.F32, tensor.MustShape(2*inner, tokens))
		modulated := round(b.Add(round(b.Multiply(round(b.Multiply(round(b.RMSNormNativeMean(stream, epsilon)), weight)), scale)), shift))
		half := func(offset uint64) *tensor.Tensor {
			return b.Reshape(b.GroupSlice(fused, offset, inner, tensor.SingletonExtent, inner), inner, tokens)
		}
		up, gate := half(0), half(inner)
		bounded := func(x *tensor.Tensor, bound float32) *tensor.Tensor {
			return round(b.Scale(round(b.Tanh(round(b.Scale(x, tensor.UnitScale/bound)))), bound))
		}
		situ := round(b.Multiply(round(b.Multiply(bounded(gate, gateBound), round(b.Sigmoid(gate)))), bounded(up, hiddenBound)))
		left := [3]*tensor.Tensor{}
		for index := range left {
			left[index] = b.Input("int8_weight_"+string(rune('a'+index)), dtype.I8ConvRot, tensor.MustShape(inner, out))
		}
		products := []*tensor.Tensor{b.MulMat(left[0], modulated), b.MulMat(left[1], modulated), b.MulMat(left[2], situ)}
		outputs := slices.Clone(products)
		if keep {
			outputs = append(outputs, modulated, situ)
		}
		testcheck.NoError(t, b.Err())
		clear(census)
		got := testcheck.Must(cuda.executeWithDeviceFeeds(context.WithoutCancel(t.Context()), outputs,
			map[*tensor.Tensor]reference.Value{
				stream: patternedValue(stream.Shape, 1, 2.3, 0.1), weight: patternedValue(weight.Shape, 2, 0.9, 1),
				scale: patternedValue(scale.Shape, 3, 0.4, 1), shift: patternedValue(shift.Shape, 4, 0.7, -0.2),
				fused: patternedValue(fused.Shape, 5, 2.9, -0.2),
			},
			map[*tensor.Tensor]driver.DevicePtr{left[0]: weights[0], left[1]: weights[1], left[2]: weights[2]},
		))(t)
		var ran []string
		for id, count := range census {
			if count > 0 && id < uint64(len(kernelFunctionNames)) {
				ran = append(ran, kernelFunctionNames[id])
			}
		}
		return [][]float32{got[products[0]].Data, got[products[1]].Data, got[products[2]].Data}, ran
	}
	computed, ran := run(false)
	for _, kernel := range []string{"quantize_int8_convrot_norm_columns_f32", "quantize_int8_convrot_situ_columns_f32"} {
		if !slices.Contains(ran, kernel) {
			t.Fatalf("%s did not run where the products alone read their inputs: %v", kernel, ran)
		}
	}
	for _, kernel := range []string{"quantize_int8_convrot_columns_f32", "rms_norm_mean_affine_bf16_round_f32", "moss_situ_bf16_round_f32"} {
		if slices.Contains(ran, kernel) {
			t.Fatalf("%s ran where the products alone read their inputs: %v", kernel, ran)
		}
	}
	stored, ran := run(true)
	if !slices.Contains(ran, "quantize_int8_convrot_columns_f32") {
		t.Fatalf("the products of stored inputs quantized none: %v", ran)
	}
	for index := range computed {
		compareBits(t, computed[index], stored[index])
	}
}

// TestAGatedSumRunsTheInt8ProductItReads holds a gated residual sum of an int8 ConvRot product, as a MOSS block adds its
// output projection (its activation stored) and its down projection (its activation a SITU over one product's halves)
// to its stream, to running the product itself and writing the sum from the product's epilogue
// (applyInt8GatedAddRewrite), never storing the product, and to changing no bit against the same sums of the products
// stored, which a graph that also keeps the products as outputs does.
func TestAGatedSumRunsTheInt8ProductItReads(t *testing.T) {
	// Serial: device fixtures share the one card.
	const inner, tokens, gateBound, hiddenBound = 256, 24, 4, 25
	storage := func(seed int) []byte {
		bytes := make([]byte, inner*inner+inner*4)
		for index := range inner * inner {
			bytes[index] = byte(int8((index*61+17*seed)%255 - 127))
		}
		for row := range inner {
			binary.LittleEndian.PutUint32(bytes[inner*inner+row*4:], math.Float32bits(float32(1+(row+seed)%7)/1024))
		}
		return bytes
	}
	worker := newFixtureWorker(t)
	weights := [2]driver.DevicePtr{copyFixtureDeviceBytes(t, worker, storage(1)), copyFixtureDeviceBytes(t, worker, storage(2))}
	cuda := newFixtureExecutorWithWorker(t, worker)
	census := map[uint64]int{}
	testcheck.NoError(t, cuda.worker.Do(t.Context(), func(state *device.State) error {
		state.Trace = &device.LaunchTrace{Census: census}
		return nil
	}))
	t.Cleanup(func() {
		_ = cuda.worker.Do(context.WithoutCancel(t.Context()), func(state *device.State) error {
			state.Trace = nil
			return nil
		})
	})
	run := func(keep bool) ([][]float32, []string) {
		b := tensor.NewBuilder()
		round := b.BF16RoundNative
		stream := b.Input("stream", dtype.F32, tensor.MustShape(inner, tokens))
		attended := b.Input("attended", dtype.F32, tensor.MustShape(inner, tokens))
		fused := b.Input("fused", dtype.F32, tensor.MustShape(2*inner, tokens))
		gates := b.Input("gates", dtype.F32, tensor.MustShape(2*inner))
		half := func(offset uint64) *tensor.Tensor {
			return b.Reshape(b.GroupSlice(fused, offset, inner, tensor.SingletonExtent, inner), inner, tokens)
		}
		up, gate := half(0), half(inner)
		bounded := func(x *tensor.Tensor, bound float32) *tensor.Tensor {
			return round(b.Scale(round(b.Tanh(round(b.Scale(x, tensor.UnitScale/bound)))), bound))
		}
		situ := round(b.Multiply(round(b.Multiply(bounded(gate, gateBound), round(b.Sigmoid(gate)))), bounded(up, hiddenBound)))
		output := b.Input("output_weight", dtype.I8ConvRot, tensor.MustShape(inner, inner))
		down := b.Input("down_weight", dtype.I8ConvRot, tensor.MustShape(inner, inner))
		products := []*tensor.Tensor{b.MulMat(output, attended), b.MulMat(down, situ)}
		attentionGate, feedGate := b.FlatSlice(gates, 0, inner), b.FlatSlice(gates, inner, inner)
		hidden := round(b.Add(stream, round(b.Multiply(products[0], attentionGate))))
		sum := round(b.Add(hidden, round(b.Multiply(products[1], feedGate))))
		outputs := []*tensor.Tensor{hidden, sum}
		if keep {
			outputs = append(outputs, products...)
		}
		testcheck.NoError(t, b.Err())
		clear(census)
		got := testcheck.Must(cuda.executeWithDeviceFeeds(context.WithoutCancel(t.Context()), outputs,
			map[*tensor.Tensor]reference.Value{
				stream: patternedValue(stream.Shape, 1, 2.3, 0.1), attended: patternedValue(attended.Shape, 2, 1.1, -0.1),
				fused: patternedValue(fused.Shape, 3, 2.9, -0.2), gates: patternedValue(gates.Shape, 4, 0.6, 0.2),
			},
			map[*tensor.Tensor]driver.DevicePtr{output: weights[0], down: weights[1]},
		))(t)
		var ran []string
		for id, count := range census {
			if count > 0 && id < uint64(len(kernelFunctionNames)) {
				ran = append(ran, kernelFunctionNames[id])
			}
		}
		return [][]float32{got[hidden].Data, got[sum].Data}, ran
	}
	summed, ran := run(false)
	if !slices.Contains(ran, "scale_int8_product_gated_add_bf16_f32") || slices.Contains(ran, "gated_add_bf16_round_f32") ||
		slices.Contains(ran, "scale_int8_product_bf16_f32") {
		t.Fatalf("the gated sums of int8 products did not each run its product's epilogue as the sum: %v", ran)
	}
	stored, ran := run(true)
	if !slices.Contains(ran, "gated_add_bf16_round_f32") {
		t.Fatalf("the gated sums of stored products summed none apart: %v", ran)
	}
	for index := range summed {
		compareBits(t, summed[index], stored[index])
	}
}
