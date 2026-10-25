//go:build windows

package executor

import (
	"context"
	"maps"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// q8CellBound is how far an attention output may stand from the reference's
// on a cache of Q8_0 cells: the host and the device may round a cell's value
// one code apart, one scale of its group, so a value error of one step reaches
// the output whole, and a key error of one step moves a score by the query's
// channels times the step times the attention scale, which one softmax turns
// into at most that much of the values' range. The bound is written from
// those terms, before any measurement.
func q8CellBound(step, scale float32, width uint64, queryMaximum float32) float64 {
	return float64(step) * (1 + float64(scale)*float64(width)*float64(queryMaximum))
}

// q8CellStep is the largest rounding step among the groups of a set of
// values: each group's absolute maximum over the code range.
func q8CellStep(t *testing.T, values []float32) float32 {
	t.Helper()
	traits, ok := dtype.Q8_0.Traits()
	if !ok || len(values)%int(traits.BlockSize) != 0 {
		t.Fatal("Q8_0 cells take whole blocks")
	}
	var step float32
	for start := 0; start < len(values); start += int(traits.BlockSize) {
		var maximum float32
		for _, value := range values[start : start+int(traits.BlockSize)] {
			maximum = max(maximum, float32(math.Abs(float64(value))))
		}
		step = max(step, maximum/q8CellCodeRange)
	}
	return step
}

// q8CellCodeRange is the code a group's absolute maximum takes.
const q8CellCodeRange = float32(127)

// TestInt8KVAttentionMatchesTheF32Reference holds the Q8_0 cell kernels to
// the F32 kernels on the values the cells return: a selected attention over a
// cache the step appended to, and a prompt's attention over a page a fused
// rope-append wrote, each against the reference backend running the same
// graph on the rounded values, within q8CellBound.
func TestInt8KVAttentionMatchesTheF32Reference(t *testing.T) {
	const (
		width, keyValueHeads, queryHeads = uint64(64), uint64(2), uint64(4)
		capacity, active, added          = uint32(256), uint32(200), uint32(8)
		selected                         = uint64(32)
		scale                            = float32(0.125)
		rotaryDimensions                 = uint32(32)
		frequencyBase                    = float32(10000)
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	draws := rand.New(rand.NewPCG(31, 17))
	fill := func(count uint64) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = 2*draws.Float32() - 1
		}
		return values
	}
	cacheShape := tensor.MustShape(width, keyValueHeads, uint64(capacity))
	rowsShape := tensor.MustShape(width, keyValueHeads, uint64(added))
	queryShape := tensor.MustShape(width, queryHeads, uint64(added))
	pastKeys, pastValues := fill(width*keyValueHeads*uint64(capacity)), fill(width*keyValueHeads*uint64(capacity))
	newKeys, newValues, queries := fill(width*keyValueHeads*uint64(added)), fill(width*keyValueHeads*uint64(added)), fill(width*queryHeads*uint64(added))
	quantized := func(values []float32) ([]byte, []float32) {
		cells, err := quant.Quantize(dtype.Q8_0, values)
		if err != nil {
			t.Fatal(err)
		}
		rounded, err := quant.Dequantize(dtype.Q8_0, cells, uint64(len(values)))
		if err != nil {
			t.Fatal(err)
		}
		return cells, rounded
	}
	keyCells, keysRounded := quantized(pastKeys)
	valueCells, valuesRounded := quantized(pastValues)
	step := max(q8CellStep(t, pastKeys), q8CellStep(t, pastValues), q8CellStep(t, newKeys), q8CellStep(t, newValues))
	bound := q8CellBound(step, scale, width, 1)

	// positions are the appended tokens' coordinates, counting from the active tokens.
	positions := make([]uint32, added)
	for token := range added {
		positions[token] = active + token
	}
	// indices select, for every appended query, the same slots: the newest tokens before it and a spread of older ones.
	indices := reference.Value{Shape: tensor.MustShape(selected, uint64(added)), Data: make([]float32, selected*uint64(added))}
	for token := range added {
		for slot := range selected {
			position := uint64(active+token) - 1 - slot*4
			indices.Data[uint64(token)*selected+slot] = float32(position)
		}
	}

	type graph struct {
		output             *tensor.Tensor
		pastKey, pastValue *tensor.Tensor
		host               map[*tensor.Tensor]reference.Value
	}
	// build makes one graph over cells of the type: a selected attention over the appended cache, or a prompt's
	// attention over the page a fused rope-append wrote.
	build := func(cells dtype.Type, prompt bool) graph {
		builder := tensor.NewBuilder()
		builder.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: active, CapacityTokens: capacity})
		pastKey := builder.Input("past_key", cells, cacheShape)
		pastValue := builder.Input("past_value", cells, cacheShape)
		newKey := builder.Input("new_key", dtype.F32, rowsShape)
		newValue := builder.Input("new_value", dtype.F32, rowsShape)
		query := builder.Input("query", dtype.F32, queryShape)
		host := map[*tensor.Tensor]reference.Value{
			newKey: {Shape: rowsShape, Data: newKeys}, newValue: {Shape: rowsShape, Data: newValues}, query: {Shape: queryShape, Data: queries},
		}
		keyRows := newKey
		var output *tensor.Tensor
		if prompt {
			keyRows = builder.RoPEWithOptions(newKey, tensor.RoPEOptions{
				Layout: tensor.RoPELayoutNormal, Positions: positions,
				RotaryDimensions: rotaryDimensions, FrequencyBase: frequencyBase, FrequencyScale: 1,
			})
		}
		cacheKey := builder.AppendCache(pastKey, keyRows, 2)
		cacheValue := builder.AppendCache(pastValue, newValue, 2)
		if prompt {
			output = builder.AttentionWithOptions(query, cacheKey, cacheValue, tensor.AttentionOptions{Scale: scale, Causal: true, QueryStart: active})
		} else {
			indexInput := builder.Input("indices", dtype.F32, indices.Shape)
			host[indexInput] = indices
			output = builder.SparseAttentionWithOffset(query, cacheKey, cacheValue, indexInput, scale, true, active)
		}
		if err := builder.Err(); err != nil {
			t.Fatal(err)
		}
		return graph{output: output, pastKey: pastKey, pastValue: pastValue, host: host}
	}
	// compare runs the Q8_0 graph on the device against the reference on the rounded values, and the F32 graph on
	// the device on the unrounded values for the reading beside it.
	compare := func(t *testing.T, prompt bool) {
		t.Helper()
		ctx := context.WithoutCancel(t.Context())
		q8 := build(dtype.Q8_0, prompt)
		referenceFeeds := map[*tensor.Tensor]reference.Value{q8.pastKey: {Shape: cacheShape, Data: keysRounded}, q8.pastValue: {Shape: cacheShape, Data: valuesRounded}}
		maps.Copy(referenceFeeds, q8.host)
		want, err := reference.Execute([]*tensor.Tensor{q8.output}, referenceFeeds)
		if err != nil {
			t.Fatal(err)
		}
		keyPointer, valuePointer := copyFixtureDeviceBytes(t, worker, keyCells), copyFixtureDeviceBytes(t, worker, valueCells)
		got, err := cuda.executeWithDeviceFeeds(ctx, []*tensor.Tensor{q8.output}, q8.host, map[*tensor.Tensor]driver.DevicePtr{q8.pastKey: keyPointer, q8.pastValue: valuePointer})
		if err != nil {
			t.Fatal(err)
		}
		f32 := build(dtype.F32, prompt)
		f32.host[f32.pastKey] = reference.Value{Shape: cacheShape, Data: pastKeys}
		f32.host[f32.pastValue] = reference.Value{Shape: cacheShape, Data: pastValues}
		plain, err := cuda.executeWithDeviceFeeds(ctx, []*tensor.Tensor{f32.output}, f32.host, nil)
		if err != nil {
			t.Fatal(err)
		}
		expected, actual, unrounded := want[q8.output].Data, got[q8.output].Data, plain[f32.output].Data
		if len(expected) != len(actual) || len(actual) != len(unrounded) || len(actual) == 0 {
			t.Fatalf("output lengths differ: reference %d, device %d, unrounded %d", len(expected), len(actual), len(unrounded))
		}
		var toReference, toUnrounded float64
		for index := range actual {
			toReference = max(toReference, math.Abs(float64(actual[index]-expected[index])))
			toUnrounded = max(toUnrounded, math.Abs(float64(actual[index]-unrounded[index])))
		}
		t.Logf("Q8_0 cells against the reference on the rounded values: %.3g; against the F32 cells on the unrounded: %.3g; bound %.3g (step %.3g)", toReference, toUnrounded, bound, step)
		if toReference > bound {
			t.Fatalf("Q8_0 cell attention differs from the reference by %.3g, past the bound %.3g", toReference, bound)
		}
		if toUnrounded > 2*bound {
			t.Fatalf("Q8_0 cell attention differs from the F32 cells by %.3g, past twice the bound %.3g", toUnrounded, bound)
		}
	}
	t.Run("selected", func(t *testing.T) { compare(t, false) })
	t.Run("prompt", func(t *testing.T) { compare(t, true) })
}
