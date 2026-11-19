//go:build windows

package executor

import (
	"context"
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// registersAttentionAgreement is the least agreement, in dB of the native kernel's output over the two's difference,
// the kernel keeping its scores in registers keeps with the native BF16 kernel on the same inputs: their arithmetic
// differs only in where a row's running maximum and sum are gathered. Measured at 135.56, 137.77 and 140.82 dB over
// these geometries (2026-11-18), F32's own rounding; BF16's rounding of one value is about 48 dB.
const registersAttentionAgreement = 100.0

// TestAttentionInRegistersAgreesWithTheNativeKernel holds attention_registers_bf16_f32 (moss-attention) to the native
// BF16 kernel it stands in for: the same BF16-rounded queries, keys and values, with and without a key bias, over
// query and key counts that leave partial tiles and grouped key heads, agree to registersAttentionAgreement, each
// geometry run by the kernel it names.
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
	}{
		{"even tiles", 4, 4, 128, 256, false},
		{"partial tiles, grouped heads", 6, 2, 197, 333, false},
		{"key bias", 2, 2, 70, 129, true},
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
			var signal, noise float64
			for index := range native {
				difference := float64(registers[index]) - float64(native[index])
				signal += float64(native[index]) * float64(native[index])
				noise += difference * difference
			}
			agreement := 10 * math.Log10(signal/noise)
			t.Logf("%s: the kernel in registers agrees with the native one to %.2f dB", geometry.name, agreement)
			if !(agreement >= registersAttentionAgreement) {
				t.Errorf("the kernel in registers agrees with the native one to %.2f dB; want at least %.0f", agreement, registersAttentionAgreement)
			}
		})
	}
}
