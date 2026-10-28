//go:build windows

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

// TestPortedMechanismsFollowModelFacts holds each ported mechanism's choice to the model's facts, not to code naming a
// model (recipe-widen), on the hermetic fixtures each family's facts are written into:
//   - int8 attention cells: a runner whose every token-holding layer is an attention layer the cell rule takes (gemma-4's
//     adapter or the qwen35 family's gated attention, each holding its own keys and values at an instantiated geometry)
//     keeps them as cells, the F32 cache where it is asked to; a geometry no kernel is instantiated for, a layer that
//     shares another's K/V and a llama layer keep the F32 cache
//   - Strata's QSA layers: a block-sparse gated attention at Strata's geometry, Flash-Next's, and no other; its verify
//     windows where every gated-delta layer is at the verify kernels' geometry too (state 128, conv 4)
//   - drafting by default: a model whose file carries its own next-token layer, and no model without one
func TestPortedMechanismsFollowModelFacts(t *testing.T) {
	// Serial: device fixtures share the one card.
	requireIntegration(t)
	cudatest.Require(t)
	type facts struct {
		cells, qsa, windows, drafts bool
	}
	for _, c := range []struct {
		name    string
		path    func(t *testing.T) string
		options OpenOptions
		want    facts
	}{
		{"qwen35 at an instantiated geometry", func(t *testing.T) string {
			return writeHermeticQwen35GGUFShaped(t, "d", hermeticContext, qwen35CellsWidth, qwen35CellsHeads, qwen35CellsKVHeads)
		}, OpenOptions{}, facts{cells: true, drafts: true}},
		{"qwen35 asked for its F32 cache", func(t *testing.T) string {
			return writeHermeticQwen35GGUFShaped(t, "d", hermeticContext, qwen35CellsWidth, qwen35CellsHeads, qwen35CellsKVHeads)
		}, OpenOptions{F32Cache: true}, facts{drafts: true}},
		{"qwen35 at a geometry no kernel holds", writeHermeticQwen35GGUF, OpenOptions{}, facts{drafts: true}},
		{"gemma-4 at its instantiated geometries", func(t *testing.T) string { return testutil.HermeticGemma4CellsGGUF(t, hermeticContext) },
			OpenOptions{}, facts{cells: true}},
		{"gemma-4 sharing K/V", func(t *testing.T) string { return testutil.HermeticGemma4SharedKVGGUF(t, hermeticContext) },
			OpenOptions{}, facts{}},
		{"llama", func(t *testing.T) string { return testutil.HermeticLlamaGGUF(t, hermeticContext) }, OpenOptions{}, facts{}},
		{"qwen4exp's block-sparse attention, its gated-delta layers at Strata's verify geometry",
			func(t *testing.T) string { return writeFlashNextFixture(t, strataFlashNextWindows) }, OpenOptions{}, facts{qsa: true, windows: true}},
		{"qwen4exp's block-sparse attention, its gated-delta layers at another state width",
			func(t *testing.T) string { return writeFlashNextFixture(t, strataFlashNext) }, OpenOptions{}, facts{qsa: true}},
	} {
		t.Run(c.name, func(t *testing.T) {
			runner, err := openF32FixtureRunner(c.path(t), c.options)
			if err != nil {
				t.Fatal(err)
			}
			defer runner.Close()
			got := facts{
				cells: slices.Contains(runner.cellLayers, true), qsa: slices.Contains(runner.qsaLayers, true),
				windows: runner.qsaWindows, drafts: runner.SpeculatesByDefault(),
			}
			if got != c.want {
				t.Fatalf("the runner chose %+v, want %+v", got, c.want)
			}
		})
	}
}
