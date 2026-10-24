//go:build windows

package executor

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestGatedDeltaKernelsMatchTheComposedKernels runs Flash-Next's gated-delta preparation on the card both ways for a
// decode token and verify spans up to tensor.GatedDeltaFusedTokens: GatedDeltaConv and GatedDeltaGates against the
// transpose, concatenation, slices, SSMConv, SiLU, L2Norm, products, Softplus and Sigmoid buildGatedDeltaMixCached
// composes. The query, key and value, every window, the gate and the beta are the composed kernels' bits.
func TestGatedDeltaKernelsMatchTheComposedKernels(t *testing.T) {
	// Serial: device fixtures share the one card.
	const kernel, epsilon = 4, 1e-6
	// Flash-Next's geometry, and the hermetic model's four-wide state, whose heads take blocks of four threads.
	for _, geometry := range []struct{ width, keyHeads, valueHeads, embedding uint64 }{{128, 16, 48, 2560}, {4, 2, 2, 24}} {
		width, keyHeads, valueHeads, embedding := geometry.width, geometry.keyHeads, geometry.valueHeads, geometry.embedding
		channels := (2*keyHeads + valueHeads) * width
		for _, tokens := range []uint64{1, 2, 3, 4, tensor.GatedDeltaFusedTokens} {
			t.Run(fmt.Sprintf("width=%d/tokens=%d", width, tokens), func(t *testing.T) {
				random := rand.New(rand.NewPCG(tokens, 17))
				b := tensor.NewBuilder()
				feeds := map[*tensor.Tensor]reference.Value{}
				input := func(name string, scale float32, dims ...uint64) *tensor.Tensor {
					node := b.Input(name, dtype.F32, tensor.MustShape(dims...))
					elements, _ := node.Shape.Elements()
					data := make([]float32, elements)
					for index := range data {
						data[index] = scale * (2*random.Float32() - 1)
					}
					feeds[node] = reference.Value{Shape: node.Shape, Data: data}
					return node
				}
				state := input("state", 1, kernel-1, channels)
				qkv := input("qkv", 1, channels, tokens)
				weights := input("conv", 0.5, kernel, channels)
				normalized := input("normalized", 1, embedding, tokens)
				alpha, beta := input("alpha", 0.05, embedding, valueHeads), input("beta", 0.05, embedding, valueHeads)
				timeStep, decay := input("dt", 1, valueHeads), input("a", 1, valueHeads)
				convInput := b.Concat(state, b.Transpose2D(qkv), tensor.FirstOffset)
				convolved := b.SiLU(b.SSMConv(convInput, weights))
				keyDimension := width * keyHeads
				query := b.L2Norm(b.GroupSlice(convolved, tensor.FirstOffset, width, keyHeads, width), epsilon)
				key := b.L2Norm(b.GroupSlice(convolved, keyDimension, width, keyHeads, width), epsilon)
				value := b.GroupSlice(convolved, 2*keyDimension, width, valueHeads, width)
				composedBeta := b.Sigmoid(b.MulMat(beta, normalized))
				composedGate := b.Multiply(b.Softplus(b.Add(b.MulMat(alpha, normalized), timeStep)), decay)
				conv := b.GatedDeltaConv(state, qkv, weights, uint32(width), uint32(keyHeads), uint32(valueHeads), epsilon)
				gates := b.GatedDeltaGates(normalized, alpha, beta, timeStep, decay)
				outputs := []*tensor.Tensor{query, key, value, composedBeta, composedGate, conv, gates}
				// The window after each token, as the composed graph slices it from the convolution's input.
				var afters []*tensor.Tensor
				for after := range tokens {
					afters = append(afters, b.GroupSlice(convInput, after+1, kernel-1, tensor.SingletonExtent, kernel-1))
				}
				outputs = append(outputs, afters...)
				if err := b.Err(); err != nil {
					t.Fatal(err)
				}
				got, err := newFixtureExecutor(t).Execute(context.WithoutCancel(t.Context()), outputs, feeds)
				if err != nil {
					t.Fatal(err)
				}
				layout := tensor.GatedDeltaConvLayout{StateWidth: width, KeyHeads: keyHeads, ValueHeads: valueHeads, Kernel: kernel, Tokens: tokens}
				packed, gated := got[conv].Data, got[gates].Data
				compareBits(t, packed[:layout.Key()], got[query].Data)
				compareBits(t, packed[layout.Key():layout.Value()], got[key].Data)
				compareBits(t, packed[layout.Value():layout.Window(0)], got[value].Data)
				for after, window := range afters {
					start := layout.Window(uint64(after))
					compareBits(t, packed[start:start+(kernel-1)*channels], got[window].Data)
				}
				compareBits(t, gated[:valueHeads*tokens], got[composedGate].Data)
				compareBits(t, gated[valueHeads*tokens:], got[composedBeta].Data)
			})
		}
	}
}

// TestGatedDeltaPromptConvMatchesTheComposedKernels runs a prompt chunk's gated-delta convolution on the card both
// ways, past tensor.GatedDeltaFusedTokens: GatedDeltaPromptConv against the transpose, concatenation, slices, SSMConv,
// SiLU and L2Norm buildGatedDeltaMixCached composes. The query, key and value and the window after the last token are
// the composed kernels' bits, at fewer tokens than the window too.
func TestGatedDeltaPromptConvMatchesTheComposedKernels(t *testing.T) {
	// Serial: device fixtures share the one card.
	const kernel, epsilon = 4, 1e-6
	for _, geometry := range []struct{ width, keyHeads, valueHeads uint64 }{{128, 16, 48}, {4, 2, 2}} {
		width, keyHeads, valueHeads := geometry.width, geometry.keyHeads, geometry.valueHeads
		channels := (2*keyHeads + valueHeads) * width
		for _, tokens := range []uint64{2, tensor.GatedDeltaFusedTokens + 1, 300, 2051} {
			t.Run(fmt.Sprintf("width=%d/tokens=%d", width, tokens), func(t *testing.T) {
				random := rand.New(rand.NewPCG(tokens, 23))
				b := tensor.NewBuilder()
				feeds := map[*tensor.Tensor]reference.Value{}
				input := func(name string, scale float32, dims ...uint64) *tensor.Tensor {
					node := b.Input(name, dtype.F32, tensor.MustShape(dims...))
					elements, _ := node.Shape.Elements()
					data := make([]float32, elements)
					for index := range data {
						data[index] = scale * (2*random.Float32() - 1)
					}
					feeds[node] = reference.Value{Shape: node.Shape, Data: data}
					return node
				}
				state := input("state", 1, kernel-1, channels)
				qkv := input("qkv", 1, channels, tokens)
				weights := input("conv", 0.5, kernel, channels)
				convInput := b.Concat(state, b.Transpose2D(qkv), tensor.FirstOffset)
				convolved := b.SiLU(b.SSMConv(convInput, weights))
				keyDimension := width * keyHeads
				query := b.L2Norm(b.GroupSlice(convolved, tensor.FirstOffset, width, keyHeads, width), epsilon)
				key := b.L2Norm(b.GroupSlice(convolved, keyDimension, width, keyHeads, width), epsilon)
				value := b.GroupSlice(convolved, 2*keyDimension, width, valueHeads, width)
				last := b.GroupSlice(convInput, tokens, kernel-1, tensor.SingletonExtent, kernel-1)
				conv := b.GatedDeltaPromptConv(state, qkv, weights, uint32(width), uint32(keyHeads), uint32(valueHeads), epsilon)
				if err := b.Err(); err != nil {
					t.Fatal(err)
				}
				got, err := newFixtureExecutor(t).Execute(context.WithoutCancel(t.Context()), []*tensor.Tensor{query, key, value, last, conv}, feeds)
				if err != nil {
					t.Fatal(err)
				}
				layout := tensor.GatedDeltaConvLayout{StateWidth: width, KeyHeads: keyHeads, ValueHeads: valueHeads, Kernel: kernel, Tokens: tokens, LastWindow: true}
				packed := got[conv].Data
				if uint64(len(packed)) != layout.Elements() {
					t.Fatalf("the prompt convolution holds %d values, want %d", len(packed), layout.Elements())
				}
				compareBits(t, packed[:layout.Key()], got[query].Data)
				compareBits(t, packed[layout.Key():layout.Value()], got[key].Data)
				compareBits(t, packed[layout.Value():layout.Window(tokens-1)], got[value].Data)
				compareBits(t, packed[layout.Window(tokens-1):], got[last].Data)
			})
		}
	}
}
