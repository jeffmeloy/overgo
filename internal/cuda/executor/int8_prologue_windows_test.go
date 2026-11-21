//go:build windows

package executor

import (
	"context"
	"encoding/binary"
	"maps"
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

// int8WeightBytes is an int8 ConvRot weight of out rows of inner patterned values, seeded, its F32 row scales after its
// rows.
func int8WeightBytes(seed, inner, out int) []byte {
	bytes := make([]byte, out*inner+out*4)
	for entry := range out * inner {
		bytes[entry] = byte(int8((entry*61+17*seed)%255 - 127))
	}
	for row := range out {
		binary.LittleEndian.PutUint32(bytes[out*inner+row*4:], math.Float32bits(float32(1+(row+seed)%7)/1024))
	}
	return bytes
}

// int8Fixture puts count int8 ConvRot weights of out rows of inner values on a fixture worker, each weight's F32 row
// scales after its rows, and gives an executor on that worker whose launches census counts by kernel, cleared before
// each graph run reads it (ranKernels).
func int8Fixture(t *testing.T, count int, inner, out int) ([]driver.DevicePtr, *Executor, map[string]int) {
	t.Helper()
	worker := newFixtureWorker(t)
	weights := make([]driver.DevicePtr, count)
	for index := range weights {
		weights[index] = copyFixtureDeviceBytes(t, worker, int8WeightBytes(index+1, inner, out))
	}
	cuda := newFixtureExecutorWithWorker(t, worker)
	census := map[string]int{}
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
	return weights, cuda, census
}

// ranKernels runs outputs of a graph on the fixture's executor from feeds and its int8 weights, and gives each output's
// values and the kernels the run launched.
func ranKernels(t *testing.T, cuda *Executor, census map[string]int, outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value,
	weights map[*tensor.Tensor]driver.DevicePtr) (map[*tensor.Tensor]reference.Value, []string) {
	t.Helper()
	clear(census)
	got := testcheck.Must(cuda.executeWithDeviceFeeds(context.WithoutCancel(t.Context()), outputs, feeds, weights))(t)
	return got, slices.Collect(maps.Keys(census))
}

// TestAnInt8ProductComputesItsNormedOrActivatedInput holds the int8 ConvRot products a MOSS block makes of a norm with
// its steps (two products of one, as q and k read one modulated norm) and of MOSS's SiTU of one product's halves as
// comfy_kitchen's quantizer computes it (tensor.Builder.MOSSSituHalves, as an int8 feed-down reads it) to quantizing
// that input as they compute it (applyInt8PrologueRewrite), never storing it, and to changing no bit against the same
// products of the input stored, which a graph that also keeps the input as an output does.
func TestAnInt8ProductComputesItsNormedOrActivatedInput(t *testing.T) {
	// Serial: device fixtures share the one card.
	const inner, out, tokens, epsilon = 256, 64, 24, 1e-6
	weights, cuda, census := int8Fixture(t, 3, inner, out)
	run := func(keep bool) ([][]float32, []string) {
		b := tensor.NewBuilder()
		round := b.BF16RoundNative
		stream := b.Input("stream", dtype.F32, tensor.MustShape(inner, tokens))
		weight := b.Input("weight", dtype.F32, tensor.MustShape(inner))
		scale := b.Input("scale", dtype.F32, tensor.MustShape(inner))
		shift := b.Input("shift", dtype.F32, tensor.MustShape(inner))
		fused := b.Input("fused", dtype.F32, tensor.MustShape(2*inner, tokens))
		modulated := round(b.Add(round(b.Multiply(round(b.Multiply(round(b.RMSNormNativeMean(stream, epsilon)), weight)), scale)), shift))
		situ := b.MOSSSituHalves(fused)
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
		got, ran := ranKernels(t, cuda, census, outputs, map[*tensor.Tensor]reference.Value{
			stream: patternedValue(stream.Shape, 1, 2.3, 0.1), weight: patternedValue(weight.Shape, 2, 0.9, 1),
			scale: patternedValue(scale.Shape, 3, 0.4, 1), shift: patternedValue(shift.Shape, 4, 0.7, -0.2),
			fused: patternedValue(fused.Shape, 5, 2.9, -0.2),
		}, map[*tensor.Tensor]driver.DevicePtr{left[0]: weights[0], left[1]: weights[1], left[2]: weights[2]})
		return [][]float32{got[products[0]].Data, got[products[1]].Data, got[products[2]].Data}, ran
	}
	computed, ran := run(false)
	for _, kernel := range []string{"quantize_int8_convrot_norm_columns_f32", "quantize_int8_convrot_situ_columns_f32"} {
		if !slices.Contains(ran, kernel) {
			t.Fatalf("%s did not run where the products alone read their inputs: %v", kernel, ran)
		}
	}
	for _, kernel := range []string{"quantize_int8_convrot_columns_f32", "rms_norm_mean_affine_bf16_round_f32", "moss_situ_halves_f32"} {
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

// TestTheSiTUOfAProductsHalvesIsItsReference holds the SiTU of a product's halves as comfy_kitchen's int8 quantizer
// computes it (moss_situ_halves_f32, tensor.Builder.MOSSSituHalves) to its reference in float64 within F32 projection
// accuracy, over values whose tanh and sigmoid saturate and values near zero, where stable_tanh keeps its precision.
func TestTheSiTUOfAProductsHalvesIsItsReference(t *testing.T) {
	// Serial: device fixtures share the one card.
	const width, rows = 96, 7
	b := tensor.NewBuilder()
	fused := b.Input("fused", dtype.F32, tensor.MustShape(2*width, rows))
	situ := b.MOSSSituHalves(fused)
	testcheck.NoError(t, b.Err())
	values := patternedValue(fused.Shape, 3, 9.7, -0.4)
	for index := range values.Data {
		if index%5 == 0 {
			values.Data[index] *= 1e-4
		}
	}
	checkCUDAGraph(t, map[*tensor.Tensor]reference.Value{fused: values}, graphOutputCheck{output: situ, tolerance: accuracyProjection})
}

// TestAGatedSumRunsTheInt8ProductItReads holds a gated residual sum of an int8 ConvRot product, as a MOSS block adds its
// output projection (its activation stored) and its down projection (its activation MOSS's SiTU of one product's
// halves, as an int8 feed-down reads it) to its stream, to running the product itself and writing the sum from the
// product's epilogue (applyInt8GatedAddRewrite), never storing the product, and to changing no bit against the same
// sums of the products stored, which a graph that also keeps the products as outputs does.
func TestAGatedSumRunsTheInt8ProductItReads(t *testing.T) {
	// Serial: device fixtures share the one card.
	const inner, tokens = 256, 24
	weights, cuda, census := int8Fixture(t, 2, inner, inner)
	run := func(keep bool) ([][]float32, []string) {
		b := tensor.NewBuilder()
		round := b.BF16RoundNative
		stream := b.Input("stream", dtype.F32, tensor.MustShape(inner, tokens))
		attended := b.Input("attended", dtype.F32, tensor.MustShape(inner, tokens))
		fused := b.Input("fused", dtype.F32, tensor.MustShape(2*inner, tokens))
		gates := b.Input("gates", dtype.F32, tensor.MustShape(2*inner))
		situ := b.MOSSSituHalves(fused)
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
		got, ran := ranKernels(t, cuda, census, outputs, map[*tensor.Tensor]reference.Value{
			stream: patternedValue(stream.Shape, 1, 2.3, 0.1), attended: patternedValue(attended.Shape, 2, 1.1, -0.1),
			fused: patternedValue(fused.Shape, 3, 2.9, -0.2), gates: patternedValue(gates.Shape, 4, 0.6, 0.2),
		}, map[*tensor.Tensor]driver.DevicePtr{output: weights[0], down: weights[1]})
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

// TestAnInt8ProductNormsAndRotatesTheHeadsItAloneGives holds a MOSS block's q and k chains, an int8 ConvRot projection of
// the block's modulated norm whose heads are normalized with their weight and rotated, to running each product's
// epilogue as the norm and rotation (applyInt8NormRotaryRewrite), never storing the product's heads, and to changing no
// bit against the same chains with the heads stored, which a graph that also keeps the heads as outputs does.
func TestAnInt8ProductNormsAndRotatesTheHeadsItAloneGives(t *testing.T) {
	// Serial: device fixtures share the one card.
	const inner, head, heads, tokens, epsilon = 256, 128, 2, 24, 1e-6
	weights, cuda, census := int8Fixture(t, 2, inner, head*heads)
	run := func(keep bool) ([][]float32, []string) {
		b := tensor.NewBuilder()
		round := b.BF16RoundNative
		stream := b.Input("stream", dtype.F32, tensor.MustShape(inner, tokens))
		feeds := map[*tensor.Tensor]reference.Value{stream: patternedValue(stream.Shape, 1, 2.3, 0.1)}
		input := func(name string, seed int, scale, bias float32, dimensions ...uint64) *tensor.Tensor {
			node := b.Input(name, dtype.F32, tensor.MustShape(dimensions...))
			feeds[node] = patternedValue(node.Shape, seed, scale, bias)
			return node
		}
		modulated := round(b.Add(round(b.Multiply(round(b.Multiply(round(b.RMSNormNativeMean(stream, epsilon)), input("norm1", 2, 0.9, 1, inner))),
			input("scale", 3, 0.4, 1, inner))), input("shift", 4, 0.7, -0.2, inner)))
		cosine, sine := input("cosine", 5, 0.11, 0.02, head, 1, tokens), input("sine", 6, 0.09, -0.01, head, 1, tokens)
		left := [2]*tensor.Tensor{}
		var rotated, kept []*tensor.Tensor
		for index := range left {
			left[index] = b.Input("int8_weight_"+string(rune('a'+index)), dtype.I8ConvRot, tensor.MustShape(inner, head*heads))
			projected := b.Reshape(b.MulMat(left[index], modulated), head, heads, tokens)
			normed := round(b.Multiply(round(b.RMSNormNativeMean(projected, epsilon)), input("norm"+string(rune('a'+index)), 7+index, 0.5, 1, head)))
			part := func(start uint64) *tensor.Tensor {
				return b.Reshape(b.GroupSlice(normed, start, head/2, tensor.SingletonExtent, head/2), head/2, heads, tokens)
			}
			turned := b.Concat(b.Scale(part(head/2), -tensor.UnitScale), part(0), tensor.FirstOffset)
			rotated = append(rotated, round(b.Add(round(b.Multiply(normed, cosine)), round(b.Multiply(turned, sine)))))
			kept = append(kept, projected)
		}
		outputs := slices.Clone(rotated)
		if keep {
			outputs = append(outputs, kept...)
		}
		testcheck.NoError(t, b.Err())
		got, ran := ranKernels(t, cuda, census, outputs, feeds, map[*tensor.Tensor]driver.DevicePtr{left[0]: weights[0], left[1]: weights[1]})
		return [][]float32{got[rotated[0]].Data, got[rotated[1]].Data}, ran
	}
	rotated, ran := run(false)
	if !slices.Contains(ran, "scale_int8_product_norm_rotate_bf16_f32") || slices.Contains(ran, "moss_norm_rotate_bf16_round_f32") ||
		slices.Contains(ran, "scale_int8_product_bf16_f32") {
		t.Fatalf("the q and k chains did not each run its product's epilogue as its norm and rotation: %v", ran)
	}
	stored, ran := run(true)
	if !slices.Contains(ran, "moss_norm_rotate_bf16_round_f32") {
		t.Fatalf("the chains of stored heads normalized and rotated none apart: %v", ran)
	}
	for index := range rotated {
		compareBits(t, rotated[index], stored[index])
	}
}
