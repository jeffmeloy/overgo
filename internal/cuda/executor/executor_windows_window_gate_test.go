//go:build windows

package executor

import (
	"fmt"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestGatedDeltaWindowStepGatesBySiLU holds a gated-delta verify window's recurrence to the host's, its outputs normed
// and gated by SiLU(z) as the qwen35 family's gated-delta layers gate them, and by sigmoid(z) as Flash-Next's, over a
// window of three tokens of two value heads a key head, every output from the state it leaves (the kept form runs the
// same gate).
func TestGatedDeltaWindowStepGatesBySiLU(t *testing.T) {
	// Serial: device fixtures share the one card.
	const keyHeads, valueHeads, tokens, epsilon = 1, 2, 3, 1e-6
	width := uint64(tensor.GatedDeltaWindowWidth)
	channels := (2*keyHeads + valueHeads) * width
	for _, silu := range []bool{true, false} {
		t.Run(fmt.Sprintf("silu=%t", silu), func(t *testing.T) {
			builder := tensor.NewBuilder()
			feeds := map[*tensor.Tensor]reference.Value{}
			input := func(name string, seed int, scale, bias float32, dims ...uint64) *tensor.Tensor {
				node := builder.Input(name, dtype.F32, tensor.MustShape(dims...))
				feeds[node] = patternedValue(node.Shape, seed, scale, bias)
				return node
			}
			state := input("state", 3, 0.01, 0, width, width, valueHeads)
			rows := input("rows", 5, 0.02, 0, channels, tokens)
			gate := input("gate", 7, 0.05, -0.4, valueHeads, tokens)
			beta := input("beta", 11, 0.02, 0.5, valueHeads, tokens)
			z := input("z", 13, 0.2, 0.1, width, valueHeads, tokens)
			gamma := input("gamma", 17, 0.03, 1, width)
			output := builder.GatedDeltaWindowStep(state, rows, gate, beta, z, gamma, nil, keyHeads, valueHeads, epsilon, silu)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			checkCUDAGraph(t, feeds, graphOutputCheck{output: output, tolerance: 2e-5})
		})
	}
}
