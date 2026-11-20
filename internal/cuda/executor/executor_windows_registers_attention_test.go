//go:build windows

package executor

import (
	"context"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestAttentionInRegistersAgreesWithTheNativeKernel holds attention_registers_bf16_f32 (moss-attention) to the native
// BF16 kernel it stands in for, bit for bit: the same BF16-rounded queries, keys and values, with and without a key
// bias, over query and key counts that leave partial tiles and grouped key heads, and random heads at MOSS's size
// whose scores spread as real ones do, each geometry run by the kernel it names. The native kernel is PyTorch's
// memory-efficient attention's arithmetic, which MOSS's frozen checkpoints hold its sampling to, and a kernel that
// differed from it at F32's rounding of a row's sum (the native kernel's to 116 dB on random heads) put a quarter of
// MOSS's first prediction outside them over a pass's 42 layers (moss-checkpoint-regression). The patterned values
// repeat every 19, so every key tile reaches its row's maximum: a maximum taken per 64-key tile agreed with the native
// kernel's per 128-key panel to 135 dB on them and to 82 dB on random heads.
func TestAttentionInRegistersAgreesWithTheNativeKernel(t *testing.T) {
	cuda := newFixtureExecutor(t)
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
	const width = 128
	for _, geometry := range []struct {
		name                   string
		queryHeads, keyHeads   uint64
		queryTokens, keyTokens uint64
		keyBias                bool
		// spread, where set, draws the queries and keys at random with that deviation, their scores spreading by its
		// square, and the values with deviation one.
		spread float64
	}{
		{"even tiles", 4, 4, 128, 256, false, 0},
		{"partial tiles, grouped heads", 6, 2, 197, 333, false, 0},
		{"key bias", 2, 2, 70, 129, true, 0},
		{"random heads at MOSS's size", 24, 24, 4118, 4118, false, 2},
	} {
		t.Run(geometry.name, func(t *testing.T) {
			run := func(registers bool) ([]float32, string) {
				builder := tensor.NewBuilder()
				query := builder.Input("q", dtype.F32, tensor.MustShape(width, geometry.queryHeads, geometry.queryTokens))
				key := builder.Input("k", dtype.F32, tensor.MustShape(width, geometry.keyHeads, geometry.keyTokens))
				value := builder.Input("v", dtype.F32, tensor.MustShape(width, geometry.keyHeads, geometry.keyTokens))
				feeds := map[*tensor.Tensor]reference.Value{
					query: patternedValue(query.Shape, 1, 0.05, 0), key: patternedValue(key.Shape, 2, 0.05, 0), value: patternedValue(value.Shape, 3, 1, 0),
				}
				if geometry.spread > 0 {
					feeds[query], feeds[key], feeds[value] = randomValue(query.Shape, 1, geometry.spread), randomValue(key.Shape, 2, geometry.spread), randomValue(value.Shape, 3, 1)
				}
				options := tensor.AttentionOptions{Scale: 1 / float32(math.Sqrt(width)), ScoresInRegisters: registers}
				if geometry.keyBias {
					bias := builder.Input("bias", dtype.F32, tensor.MustShape(geometry.keyTokens))
					feeds[bias] = patternedValue(bias.Shape, 4, 2, -1)
					options.KeyBias = bias
				}
				attention := builder.AttentionWithOptions(builder.BF16RoundNative(query), builder.BF16RoundNative(key), builder.BF16RoundNative(value), options)
				testcheck.NoError(t, builder.Err())
				clear(census)
				got := testcheck.Must(cuda.Execute(context.WithoutCancel(t.Context()), []*tensor.Tensor{attention}, feeds))(t)
				kernel := "attention_native_bf16_f32"
				if registers {
					kernel = "attention_registers_bf16_f32"
				}
				if at := slices.Index(kernelFunctionNames[:], kernel); at < 0 || census[uint64(at)] == 0 {
					t.Fatalf("%s did not run: %v", kernel, census)
				}
				return got[attention].Data, kernel
			}
			native, _ := run(false)
			registers, _ := run(true)
			compareBits(t, registers, native)
		})
	}
}

// randomValue is shape's values drawn from a normal distribution of deviation spread, the same for the same seed.
func randomValue(shape tensor.Shape, seed uint64, spread float64) reference.Value {
	elements, _ := shape.Elements()
	draws := rand.New(rand.NewPCG(seed, elements))
	data := make([]float32, int(elements))
	for index := range data {
		data[index] = float32(draws.NormFloat64() * spread)
	}
	value, _ := reference.NewValue(shape, data)
	return value
}
