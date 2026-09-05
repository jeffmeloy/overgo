//go:build windows

package executor

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestDeviceBufferPoolReleaseBound reproduces a returned batch exceeding the
// existing free limit without a subsequent allocation to trigger trimming.
func TestDeviceBufferPoolReleaseBound(t *testing.T) {
	for _, retained := range []bool{false, true} {
		name := "device buffers"
		if retained {
			name = "retained graph outputs"
		}
		t.Run(name, func(t *testing.T) {
			cuda := newFixtureExecutor(t)
			// Two aligned buffers fit; a batch of three must release one.
			limit := 2 * deviceAllocationAlignment
			if err := cuda.worker.Do(t.Context(), func(*device.State) error {
				cuda.resources.buffers.freeLimit = limit
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			builder := tensor.NewBuilder()
			shape := tensor.MustShape(4)
			input := builder.Input("input", dtype.F32, shape)
			output := builder.Scale(input, 2)
			value, err := reference.NewValue(shape, []float32{1, 2, 3, 4})
			if err != nil {
				t.Fatal(err)
			}
			var warm uint64
			for cycle, count := range []int{3, 1, 4, 1, 4} {
				var buffers []*DeviceBuffer
				var outputs []*RetainedOutputs
				for range count {
					if retained {
						result, err := cuda.executeRetainedWithDeviceFeeds(t.Context(), []*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{input: value}, nil)
						if err != nil {
							t.Fatal(err)
						}
						got, err := result.CopyToHost(t.Context(), output)
						if err != nil {
							t.Fatal(err)
						}
						compare(t, got.Data, []float32{2, 4, 6, 8}, accuracyExact)
						outputs = append(outputs, result)
						continue
					}
					buffer, err := cuda.AllocateDeviceBuffer(t.Context(), deviceAllocationAlignment)
					if err != nil {
						t.Fatal(err)
					}
					buffers = append(buffers, buffer)
				}
				canceled, cancel := context.WithCancelCause(t.Context())
				cancel(context.Canceled)
				if retained {
					if err := outputs[0].Release(canceled); !errors.Is(err, context.Canceled) {
						t.Fatalf("canceled release: %v", err)
					}
					for _, result := range outputs {
						for range 2 { // A retry after release must not return the same lease twice.
							if err := result.Release(t.Context()); err != nil {
								t.Fatal(err)
							}
						}
					}
				} else {
					if err := ReleaseDeviceBuffers(canceled, buffers...); !errors.Is(err, context.Canceled) {
						t.Fatalf("canceled batch release: %v", err)
					}
					for range 2 {
						if err := ReleaseDeviceBuffers(t.Context(), buffers...); err != nil {
							t.Fatal(err)
						}
					}
				}
				if err := cuda.worker.Do(t.Context(), func(*device.State) error {
					pool := &cuda.resources.buffers
					var allocated uint64
					for _, lease := range pool.allocations {
						allocated += lease.size
					}
					if allocated != pool.freeBytes || pool.freeBytes > limit {
						t.Errorf("cycle %d returned pool: allocated=%d free=%d limit=%d", cycle, allocated, pool.freeBytes, limit)
					}
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				memory, err := cuda.worker.MemoryStats(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				if cycle == 0 {
					warm = memory.CurrentBytes
				} else if memory.CurrentBytes > warm {
					t.Fatalf("cycle %d retained bytes grew from %d to %d", cycle, warm, memory.CurrentBytes)
				}
			}
		})
	}
	t.Run("failed graph releases temporary and retained buffers", func(t *testing.T) {
		cuda := newFixtureExecutor(t)
		limit := 2 * deviceAllocationAlignment
		if err := cuda.worker.Do(t.Context(), func(*device.State) error {
			cuda.resources.buffers.freeLimit = limit
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		builder := tensor.NewBuilder()
		shape := tensor.MustShape(limit)
		left := builder.Input("left", dtype.F32, shape)
		right := builder.Input("missing", dtype.F32, shape)
		output := builder.Add(left, right)
		compiled, err := Compile(output)
		if err != nil {
			t.Fatal(err)
		}
		value, err := reference.NewValue(shape, make([]float32, limit))
		if err != nil {
			t.Fatal(err)
		}
		feeds := map[*tensor.Tensor]reference.Value{left: value}
		for _, retain := range []bool{false, true} {
			_, err := cuda.runCompiled(t.Context(), compiled, feeds, nil, nil, nil, retain, false)
			if err == nil || !strings.Contains(err.Error(), "missing") {
				t.Fatalf("missing feed: %v", err)
			}
			if err := cuda.worker.Do(t.Context(), func(*device.State) error {
				pool := &cuda.resources.buffers
				if len(pool.allocations) != 0 || pool.freeBytes != 0 {
					t.Errorf("failed graph retained allocations=%d free=%d", len(pool.allocations), pool.freeBytes)
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		}
	})
}
