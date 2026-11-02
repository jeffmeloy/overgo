//go:build windows

package executor

import (
	"cmp"
	"context"
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// nanAttentionCase is a geometry that routes attention to one kernel: past tokens with a capacity take the decode kernels,
// and selected slots the sparse ones.
type nanAttentionCase struct {
	name                        string
	kernel                      string
	width, queryHeads, keyHeads uint64
	queryTokens, keyTokens      uint64
	pastTokens, capacity        uint64
	selected                    uint64
	causal, bf16                bool
	tensorCores, memorySaving   bool
	scale                       float32
}

// build returns the case's attention, its query input as the graph takes it, and every input's values.
func (c nanAttentionCase) build(t *testing.T) (*tensor.Tensor, *tensor.Tensor, map[*tensor.Tensor]reference.Value) {
	t.Helper()
	const inputScale = 0.01
	builder := tensor.NewBuilder()
	query := builder.Input("q", dtype.F32, tensor.MustShape(c.width, c.queryHeads, c.queryTokens))
	feeds := map[*tensor.Tensor]reference.Value{query: patternedValue(query.Shape, 1, inputScale, 0)}
	var key, value *tensor.Tensor
	if c.capacity != 0 {
		builder.SetCacheAppendPlan(tensor.CacheAppendPlan{CapacityTokens: uint32(c.capacity), ActiveTokens: uint32(c.pastTokens)})
		pastKey := builder.Input("past_k", dtype.F32, tensor.MustShape(c.width, c.keyHeads, c.capacity))
		newKey := builder.Input("new_k", dtype.F32, tensor.MustShape(c.width, c.keyHeads, c.queryTokens))
		pastValue := builder.Input("past_v", dtype.F32, tensor.MustShape(c.width, c.keyHeads, c.capacity))
		newValue := builder.Input("new_v", dtype.F32, tensor.MustShape(c.width, c.keyHeads, c.queryTokens))
		key, value = builder.AppendCache(pastKey, newKey, 2), builder.AppendCache(pastValue, newValue, 2)
		feeds[pastKey], feeds[newKey] = patternedValue(pastKey.Shape, 2, inputScale, 0), patternedValue(newKey.Shape, 4, inputScale, 0)
		feeds[pastValue], feeds[newValue] = patternedValue(pastValue.Shape, 3, 1, 0), patternedValue(newValue.Shape, 5, 1, 0)
	} else {
		key = builder.Input("k", dtype.F32, tensor.MustShape(c.width, c.keyHeads, c.keyTokens))
		value = builder.Input("v", dtype.F32, tensor.MustShape(c.width, c.keyHeads, c.keyTokens))
		feeds[key], feeds[value] = patternedValue(key.Shape, 2, inputScale, 0), patternedValue(value.Shape, 3, 1, 0)
	}
	queried := query
	if c.bf16 {
		round := builder.BF16Round
		if c.width < uint64(attentionBF16Width) {
			round = builder.BF16RoundNative
		}
		queried, key, value = round(query), round(key), round(value)
	}
	scale := cmp.Or(c.scale, 1/float32(math.Sqrt(float64(c.width))))
	var attention *tensor.Tensor
	if c.selected != 0 {
		slots := builder.Input("indices", dtype.F32, tensor.MustShape(c.selected, c.queryTokens))
		indices := make([]float32, c.selected*c.queryTokens)
		for index := range indices {
			indices[index] = float32(uint64(index) * 7 % c.pastTokens)
		}
		feeds[slots] = reference.Value{Shape: slots.Shape, Data: indices}
		attention = builder.SparseAttentionWithOffset(queried, key, value, slots, scale, true, uint32(c.pastTokens))
	} else {
		attention = builder.AttentionWithOptions(queried, key, value, tensor.AttentionOptions{
			Scale: scale, Causal: c.causal, QueryStart: uint32(c.pastTokens),
			TensorCores: c.tensorCores, MemoryEfficientF32: c.memorySaving,
		})
	}
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	return attention, query, feeds
}

// nanAttentionCases are the geometries that reach each attention kernel.
var nanAttentionCases = []nanAttentionCase{
	{name: "tiled", kernel: "attention_tiled_f32", width: 384, queryHeads: 1, keyHeads: 1, queryTokens: 257, keyTokens: 257, memorySaving: true},
	{name: "online", kernel: "attention_online_finalize_f32", width: 96, queryHeads: 4, keyHeads: 2, queryTokens: 100, keyTokens: 77},
	{name: "tiled_bf16", kernel: "attention_tiled_bf16_f32", width: 128, queryHeads: 4, keyHeads: 4, queryTokens: 128, keyTokens: 96, bf16: true},
	{name: "native_bf16", kernel: "attention_native_bf16_f32", width: 64, queryHeads: 4, keyHeads: 2, queryTokens: 65, keyTokens: 129, bf16: true},
	{name: "flash_64", kernel: "attention_flash_64_f32", width: 64, queryHeads: 4, keyHeads: 4, queryTokens: 96, keyTokens: 96, causal: true, tensorCores: true},
	{name: "flash_128", kernel: "attention_flash_128_f32", width: 128, queryHeads: 16, keyHeads: 2, queryTokens: 70, keyTokens: 70, causal: true, tensorCores: true},
	{name: "flash_256", kernel: "attention_flash_256_f32", width: 256, queryHeads: 8, keyHeads: 2, queryTokens: 100, keyTokens: 100, causal: true, tensorCores: true},
	{name: "decode", kernel: "attention_decode_f32", width: 4, queryHeads: 2, keyHeads: 1, queryTokens: 1, pastTokens: 5, capacity: 8, causal: true, scale: 0.5},
	{name: "decode_combine", kernel: "attention_decode_combine_f32", width: 4, queryHeads: 2, keyHeads: 1, queryTokens: 1, pastTokens: 16500, capacity: 20000, causal: true, scale: 0.5},
	{name: "sparse_split", kernel: "sparse_attention_split_f32", width: 256, queryHeads: 24, keyHeads: 2, queryTokens: 1, keyTokens: 3000, pastTokens: 2999, selected: 2048},
	{name: "sparse_mma", kernel: "sparse_attention_mma_f32", width: 256, queryHeads: 24, keyHeads: 2, queryTokens: 130, keyTokens: 300, pastTokens: 170, selected: 64},
}

// TestAttentionWritesEveryRowOnNaN: a NaN in one query's row reaches that row of the output in every channel, whichever
// kernel takes it, and the rows beside it stay finite, so no row keeps what the buffer held or reads as a plausible value.
// The census holds each case to the kernel it names, so a route that moves cannot leave a kernel untested.
func TestAttentionWritesEveryRowOnNaN(t *testing.T) {
	cuda := newFixtureExecutor(t)
	census := map[uint64]int{}
	if err := cuda.worker.Do(t.Context(), func(state *device.State) error {
		state.Trace = &device.LaunchTrace{Census: census}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cuda.worker.Do(context.WithoutCancel(t.Context()), func(state *device.State) error {
			state.Trace = nil
			return nil
		})
	})
	for _, c := range nanAttentionCases {
		t.Run(c.name, func(t *testing.T) {
			attention, query, feeds := c.build(t)
			poisoned := reference.Value{Shape: feeds[query].Shape, Data: slices.Clone(feeds[query].Data)}
			token, head := c.queryTokens/2, c.queryHeads-1
			poisoned.Data[(token*c.queryHeads+head)*c.width] = float32(math.NaN())
			feeds[query] = poisoned
			clear(census)
			got, err := cuda.Execute(context.WithoutCancel(t.Context()), []*tensor.Tensor{attention}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			at := slices.Index(kernelFunctionNames[:], c.kernel)
			if at < 0 || census[uint64(at)] == 0 {
				t.Fatalf("%s did not run: %v", c.kernel, census)
			}
			for row := range c.queryTokens * c.queryHeads {
				values := got[attention].Data[row*c.width : (row+1)*c.width]
				nans := 0
				for _, value := range values {
					if value != value {
						nans++
					}
				}
				want := 0
				if row == token*c.queryHeads+head {
					want = len(values)
				}
				if nans != want {
					t.Fatalf("row %d holds %d NaN channels of %d, want %d", row, nans, len(values), want)
				}
			}
		})
	}
	t.Run("strata", func(t *testing.T) { strataAttentionParity(t, cuda, true, 1500, 256) })
}
