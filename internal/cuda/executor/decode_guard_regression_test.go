//go:build windows

package executor

import (
	"fmt"
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// decodeGuardGraph uses the same resident-cache layout as the inherited
// adaptive decode_attn operator. The reference executor owns its CPU oracle.
func decodeGuardGraph(t testing.TB, width, queryHeads, keyHeads uint64, capacity, active uint32) (*tensor.Tensor, map[*tensor.Tensor]reference.Value) {
	t.Helper()
	builder := tensor.NewBuilder()
	builder.SetCacheAppendPlan(tensor.CacheAppendPlan{CapacityTokens: capacity, ActiveTokens: active})
	query := builder.Input("query", dtype.F32, tensor.MustShape(width, queryHeads, 1))
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(width, keyHeads, uint64(capacity)))
	newKey := builder.Input("new_key", dtype.F32, tensor.MustShape(width, keyHeads, 1))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(width, keyHeads, uint64(capacity)))
	newValue := builder.Input("new_value", dtype.F32, tensor.MustShape(width, keyHeads, 1))
	key := builder.AppendCache(pastKey, newKey, 2)
	value := builder.AppendCache(pastValue, newValue, 2)
	output := builder.AttentionWithOptions(query, key, value, tensor.AttentionOptions{
		Scale: float32(1 / math.Sqrt(float64(width))), Causal: true, QueryStart: active,
	})
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	return output, map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 1, 0.2, 0), pastKey: patternedValue(pastKey.Shape, 2, 0.15, 0),
		newKey: patternedValue(newKey.Shape, 3, 0.15, 0), pastValue: patternedValue(pastValue.Shape, 4, 0.2, 0),
		newValue: patternedValue(newValue.Shape, 5, 0.2, 0),
	}
}

func residentDecodeInputs(t testing.TB, cuda *Executor, compiled *CompiledGraph, feeds map[*tensor.Tensor]reference.Value) *DeviceInputs {
	t.Helper()
	inputs := compiled.NewDeviceInputs()
	for node, value := range feeds {
		pointer := copyFixtureDeviceBytes(t, cuda.worker, driver.Bytes(value.Data))
		if err := inputs.Set(node, pointer); err != nil {
			t.Fatal(err)
		}
	}
	return inputs
}

func TestDecodeGuardRegression(t *testing.T) {
	scalarBytes, ok := dtype.F32.ScalarBytes()
	if !ok {
		t.Fatal("F32 has no scalar storage width")
	}
	type cacheShape struct{ capacity, active uint32 }
	for _, geometry := range []struct{ width, queryHeads, keyHeads uint64 }{{4, 2, 1}, {64, 14, 2}, {96, 4, 2}, {128, 8, 2}, {128, 16, 2}, {192, 4, 1}, {256, 4, 1}, {320, 4, 1}} {
		shapes := []cacheShape{{256, 255}, {512, 256}, {512, 511}, {2048, 1792}, {4096, 2048}, {16384, 8448}}
		if geometry.width == 4 {
			// More than 8192 keys per split exercises scratch reuse across
			// softmax tiles; the tiny head keeps the CPU oracle inexpensive.
			shapes = []cacheShape{{262144, 131073}}
		}
		for _, shape := range shapes {
			t.Run(fmt.Sprintf("width=%d/capacity=%d/active=%d", geometry.width, shape.capacity, shape.active), func(t *testing.T) {
				output, feeds := decodeGuardGraph(t, geometry.width, geometry.queryHeads, geometry.keyHeads, shape.capacity, shape.active)
				want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
				if err != nil {
					t.Fatal(err)
				}
				compiled, err := Compile(output)
				if err != nil {
					t.Fatal(err)
				}
				cuda := newFixtureExecutor(t)
				inputs := residentDecodeInputs(t, cuda, compiled, feeds)
				before, err := cuda.worker.ExecutionStats(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				got, err := cuda.ExecuteCompiled(t.Context(), compiled, nil, inputs)
				if err != nil {
					t.Fatal(err)
				}
				after, err := cuda.worker.ExecutionStats(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				// Uploading even one capacity-sized cache invalidates resident
				// timing. Scalar runtime attributes remain legitimate traffic.
				cacheBytes := uint64(shape.capacity) * geometry.width * geometry.keyHeads * scalarBytes
				if uploaded := after.HostToDeviceBytes - before.HostToDeviceBytes; uploaded >= cacheBytes {
					t.Fatalf("resident execution uploaded %d bytes, cache=%d", uploaded, cacheBytes)
				}
				worst, at := maxAbsDifference(t, got[output].Data, want[output].Data)
				if worst > accuracyProjection {
					t.Fatalf("resident attention differs from CPU by %.6g at %d (bound %.6g)", worst, at, accuracyProjection)
				}
				t.Logf("CPU/device max_abs=%.6g upload_bytes=%d cache_bytes=%d", worst, after.HostToDeviceBytes-before.HostToDeviceBytes, cacheBytes)
			})
		}
	}
}
