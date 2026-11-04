//go:build windows

package inference

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor/dtype"
)

// TestQwen35GatedDeltaLayersHoldTheWindow holds the qwen35 family's gated-delta layers to Strata's verify window: a
// hybrid at the family's geometry, a 128-wide state over a 4-tap convolution with its alpha and beta in Q8_0, as
// Qwen3.5-9B-Q8_0 and Qwen3.8-27B-UD-Q6_K hold them, has every gated-delta layer hold the window, its gates composed
// from the quantized products and its output gated by SiLU. The hermetic hybrid's 4-wide state holds none.
func TestQwen35GatedDeltaLayersHoldTheWindow(t *testing.T) {
	// Serial: device fixtures share the one card.
	requireIntegration(t)
	cudatest.Require(t)
	family := qwen35Geometry{embedding: 64, ffn: 64, inner: 256, state: 128, groups: 1, rank: 2, gates: dtype.Q8_0}
	for _, c := range []struct {
		name  string
		path  string
		holds bool
	}{
		{"the family's geometry", writeQwen35Fixture(t, family, "d", qwen35CellsContext, qwen35CellsWidth, qwen35CellsHeads, qwen35CellsKVHeads), true},
		{"the hermetic hybrid", writeHermeticQwen35GGUF(t), false},
	} {
		runner, err := openF32FixtureRunner(c.path, OpenOptions{})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		recurrent := 0
		for layer, weights := range runner.weights.Layers {
			plan := runner.layerProgram(layer).Layer()
			if !plan.Recurrent {
				continue
			}
			recurrent++
			if holds := runner.spec.HoldsGatedDeltaWindow(plan, weights); holds != c.holds {
				t.Errorf("%s: gated-delta layer %d holds the window %t, want %t", c.name, layer, holds, c.holds)
			}
		}
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
		if recurrent == 0 {
			t.Fatalf("%s holds no gated-delta layer", c.name)
		}
	}
}
