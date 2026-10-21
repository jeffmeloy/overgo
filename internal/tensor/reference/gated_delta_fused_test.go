package reference

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestGatedDeltaOperationsMatchTheComposedPreparation: GatedDeltaConv and GatedDeltaGates give the bits of the
// preparation buildGatedDeltaMixCached composes for a separate-projection gated-delta layer, for a decode token and a
// span: the L2-normalized query and key, the value, the convolution window after each token (the next state and every
// checkpoint's), the decay gate and the beta.
func TestGatedDeltaOperationsMatchTheComposedPreparation(t *testing.T) {
	const width, keyHeads, valueHeads, kernel, embedding, epsilon = 16, 2, 6, 4, 40, 1e-6
	channels := uint64((2*keyHeads + valueHeads) * width)
	for _, tokens := range []uint64{1, 3} {
		t.Run(fmt.Sprintf("tokens=%d", tokens), func(t *testing.T) {
			random := rand.New(rand.NewPCG(tokens, 13))
			b := tensor.NewBuilder()
			feeds := map[*tensor.Tensor]Value{}
			input := func(name string, dims ...uint64) *tensor.Tensor {
				node := b.Input(name, dtype.F32, tensor.MustShape(dims...))
				elements, _ := node.Shape.Elements()
				data := make([]float32, elements)
				for index := range data {
					data[index] = 2*random.Float32() - 1
				}
				feeds[node] = Value{Shape: node.Shape, Data: data}
				return node
			}
			state := input("state", kernel-1, channels)
			qkv := input("qkv", channels, tokens)
			weights := input("conv", kernel, channels)
			normalized := input("normalized", embedding, tokens)
			alphaWeight, betaWeight := input("alpha", embedding, valueHeads), input("beta", embedding, valueHeads)
			timeStep, decay := input("dt", valueHeads), input("a", valueHeads)
			// The composed preparation, as buildGatedDeltaMixCached builds it for one sequence.
			convInput := b.Concat(state, b.Transpose2D(qkv), tensor.FirstOffset)
			convolved := b.SiLU(b.SSMConv(convInput, weights))
			keyDimension := uint64(width * keyHeads)
			query := b.L2Norm(b.GroupSlice(convolved, tensor.FirstOffset, width, keyHeads, width), epsilon)
			key := b.L2Norm(b.GroupSlice(convolved, keyDimension, width, keyHeads, width), epsilon)
			value := b.GroupSlice(convolved, 2*keyDimension, width, valueHeads, width)
			beta := b.Sigmoid(b.MulMat(betaWeight, normalized))
			gate := b.Multiply(b.Softplus(b.Add(b.MulMat(alphaWeight, normalized), timeStep)), decay)
			outputs := []*tensor.Tensor{query, key, value, beta, gate}
			var windows []*tensor.Tensor
			for after := range tokens {
				window := b.Reshape(b.GroupSlice(convInput, after+1, kernel-1, tensor.SingletonExtent, kernel-1), kernel-1, channels)
				windows = append(windows, window)
				outputs = append(outputs, window)
			}
			conv := b.GatedDeltaConv(state, qkv, weights, width, keyHeads, valueHeads, epsilon)
			gates := b.GatedDeltaGates(normalized, alphaWeight, betaWeight, timeStep, decay)
			outputs = append(outputs, conv, gates)
			if err := b.Err(); err != nil {
				t.Fatal(err)
			}
			got, err := Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			same := func(name string, fused, composed []float32) {
				t.Helper()
				if !slices.EqualFunc(fused, composed, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) }) {
					t.Fatalf("%s differs from the composed preparation: %v against %v", name, fused, composed)
				}
			}
			layout := tensor.GatedDeltaConvLayout{StateWidth: width, KeyHeads: keyHeads, ValueHeads: valueHeads, Kernel: kernel, Tokens: tokens}
			packed := got[conv].Data
			same("query", packed[:layout.Key()], got[query].Data)
			same("key", packed[layout.Key():layout.Value()], got[key].Data)
			same("value", packed[layout.Value():layout.Window(0)], got[value].Data)
			for after, window := range windows {
				start := layout.Window(uint64(after))
				same(fmt.Sprintf("window after token %d", after), packed[start:start+(kernel-1)*channels], got[window].Data)
			}
			gated := got[gates].Data
			same("gate", gated[:valueHeads*tokens], got[gate].Data)
			same("beta", gated[valueHeads*tokens:], got[beta].Data)
		})
	}
}
