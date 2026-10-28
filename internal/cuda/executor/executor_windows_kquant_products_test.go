//go:build windows

package executor

import (
	"context"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestKQuantSpansReadEachWeightOnce holds the K-quant products of one to eight columns to Strata's native MMVQ
// (kQuantProducts): through the executor, each column of a product is bitwise the native MMVQ's on that column alone,
// whatever the window's width, so a verify window's rows are the rows its tokens would decode; nine columns keep the
// q8-input span kernels, within rounding of the same products.
func TestKQuantSpansReadEachWeightOnce(t *testing.T) {
	// Serial: device fixtures share the one card.
	const (
		nIn, nOut = 2560, 300
		widest    = q8InputSpanColumns + 1
		// bound is the relative L1 distance the span kernels' nine columns keep from the native products'.
		bound = 1e-2
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(29, 31))
	for _, storage := range []dtype.Type{dtype.Q4K, dtype.Q5K, dtype.Q6K} {
		ggml := kQuantProducts[storage]
		weights := buffers.upload(t, strataRandomWeights(t, draws, storage, nIn, nOut, 0.02))
		x := make([]float32, nIn*widest)
		for index := range x {
			x[index] = float32(draws.NormFloat64())
		}
		// The native MMVQ's product of each column alone.
		alone := make([][]float32, widest)
		for column := range widest {
			xBuffer := buffers.upload(t, float32Bytes(x[column*nIn:(column+1)*nIn]))
			xq := buffers.zeros(t, int(strataNativeQ81Size(nIn, 1)))
			y := buffers.zeros(t, 4*nOut)
			buffers.run(t, func(state *device.State, functions functionSet) error {
				if err := launchStrataNativeQuantizeQ81(state, functions, xBuffer, xq, nIn, 1); err != nil {
					return err
				}
				return launchStrataNativeMMVQ(state, functions, ggml, weights, xq, y, nIn, nOut, 1)
			})
			alone[column] = float32s(buffers.read(t, y, 4*nOut))
		}
		for columns := 1; columns <= widest; columns++ {
			builder := tensor.NewBuilder()
			weightInput := builder.Input("weights", storage, tensor.MustShape(nIn, nOut))
			input := builder.Input("input", dtype.F32, tensor.MustShape(nIn, uint64(columns)))
			product := builder.MulMat(weightInput, input)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			value, err := reference.NewValue(input.Shape, x[:nIn*columns])
			if err != nil {
				t.Fatal(err)
			}
			results, err := cuda.executeWithDeviceFeeds(context.WithoutCancel(t.Context()), []*tensor.Tensor{product},
				map[*tensor.Tensor]reference.Value{input: value}, map[*tensor.Tensor]driver.DevicePtr{weightInput: weights})
			if err != nil {
				t.Fatal(err)
			}
			got := results[product].Data
			for column := range columns {
				row, want := got[column*nOut:(column+1)*nOut], alone[column]
				if columns <= q8InputSpanColumns && !slices.Equal(row, want) {
					t.Fatalf("%s: column %d of a %d-column product is not the native MMVQ's column alone", storage, column, columns)
				}
				if d := strataRelativeL1(float32To64(row), float32To64(want)); d > bound {
					t.Fatalf("%s: column %d of a %d-column product is %.3g from the native product", storage, column, columns, d)
				}
			}
		}
	}
}
