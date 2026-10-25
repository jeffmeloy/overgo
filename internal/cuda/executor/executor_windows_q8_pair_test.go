//go:build windows

package executor

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestSharedExpertFusionsMatchTheComposedKernels runs Flash-Next's gated shared expert, Add(routed, Multiply(down ·
// SwiGLU(gate · x, up · x), Sigmoid(router · x))) over Q8_0 rows, for a decode token and verify spans: fused (the gate
// and up products in one launch, the sigmoid gate and the addition in the down product's) and composed (the products
// held as outputs, so no fusion takes them). The fused graph takes both fusions and its output is the composed bits.
func TestSharedExpertFusionsMatchTheComposedKernels(t *testing.T) {
	// Serial: device fixtures share the one card.
	const width, hidden = 2560, 640
	for _, tokens := range []uint64{1, 2, 3, 4, q8InputSpanColumns} {
		t.Run(fmt.Sprintf("tokens=%d", tokens), func(t *testing.T) {
			random := rand.New(rand.NewPCG(tokens, 19))
			values := func(count uint64, scale float32) []float32 {
				data := make([]float32, count)
				for index := range data {
					data[index] = scale * (2*random.Float32() - 1)
				}
				return data
			}
			stored := func(count uint64) []byte {
				bytes, err := quant.Quantize(dtype.Q8_0, values(count, 3/float32(math.Sqrt(width))))
				if err != nil {
					t.Fatal(err)
				}
				return bytes
			}
			gateRows, upRows, downRows := stored(width*hidden), stored(width*hidden), stored(hidden*width)
			x, routed, router := values(width*tokens, 1), values(width*tokens, 1), values(width, 0.05)
			cuda := newFixtureExecutor(t)
			weights := []driver.DevicePtr{
				copyFixtureDeviceBytes(t, cuda.worker, gateRows), copyFixtureDeviceBytes(t, cuda.worker, upRows),
				copyFixtureDeviceBytes(t, cuda.worker, downRows),
			}
			run := func(composed bool) []float32 {
				b := tensor.NewBuilder()
				input := b.Input("x", dtype.F32, tensor.MustShape(width, tokens))
				routedInput := b.Input("routed", dtype.F32, tensor.MustShape(width, tokens))
				routerInput := b.Input("router", dtype.F32, tensor.MustShape(width, tensor.SingletonExtent))
				gate := b.Input("gate", dtype.Q8_0, tensor.MustShape(width, hidden))
				up := b.Input("up", dtype.Q8_0, tensor.MustShape(width, hidden))
				down := b.Input("down", dtype.Q8_0, tensor.MustShape(hidden, width))
				gateProduct, upProduct := b.MulMat(gate, input), b.MulMat(up, input)
				shared := b.Multiply(b.MulMat(down, b.SwiGLU(gateProduct, upProduct)), b.Sigmoid(b.MulMat(routerInput, input)))
				output := b.Add(routedInput, shared)
				outputs := []*tensor.Tensor{output}
				if composed {
					outputs = append(outputs, gateProduct, upProduct, shared)
				}
				if err := b.Err(); err != nil {
					t.Fatal(err)
				}
				if !composed {
					compiled, err := Compile(outputs...)
					if err != nil {
						t.Fatal(err)
					}
					for node, kind := range map[*tensor.Tensor]compiledFusionKind{output: compiledFusionQ8GatedAdd, upProduct: compiledFusionQ8Pair, gateProduct: compiledFusionQ8PairPeer} {
						if fusion := compiled.nodes[compiled.orderIndexes[node]].fusion; fusion == nil || fusion.kind != kind {
							t.Fatalf("node %d takes fusion %+v, want kind %d", node.ID, fusion, kind)
						}
					}
				}
				got, err := cuda.executeWithDeviceFeeds(context.WithoutCancel(t.Context()), outputs,
					map[*tensor.Tensor]reference.Value{
						input: {Shape: input.Shape, Data: x}, routedInput: {Shape: routedInput.Shape, Data: routed},
						routerInput: {Shape: routerInput.Shape, Data: router},
					},
					map[*tensor.Tensor]driver.DevicePtr{gate: weights[0], up: weights[1], down: weights[2]})
				if err != nil {
					t.Fatal(err)
				}
				return got[output].Data
			}
			compareBits(t, run(false), run(true))
		})
	}
}
