package sampling

import (
	"slices"
	"testing"
)

// TestMossLayerSkipIsTheGenericBlockOverride holds the block override a guider carries (internal/latentimage holds
// MOSS's side): one over the whole schedule, as MOSS's negative branch skips block 22, names its blocks at every step;
// one over part of it, as skip-layer guidance runs, names them at the steps whose fraction lies in its span and no
// others; one of no blocks names none; and a block the model lacks, or a span outside the schedule, is refused.
func TestMossLayerSkipIsTheGenericBlockOverride(t *testing.T) {
	t.Parallel()
	whole := BlockOverride{Blocks: []int{22}, Stop: 1}
	for step := range 25 {
		if !slices.Equal(whole.At(step, 25), []int{22}) {
			t.Fatalf("step %d of 25 overrides %v, want block 22", step, whole.At(step, 25))
		}
	}
	part := BlockOverride{Blocks: []int{9}, Start: 0.25, Stop: 0.5}
	var held []int
	for step := range 8 {
		if len(part.At(step, 8)) > 0 {
			held = append(held, step)
		}
	}
	if !slices.Equal(held, []int{2, 3}) || (BlockOverride{Stop: 1}).At(0, 8) != nil {
		t.Fatalf("a quarter-to-half span overrides steps %v of 8, want 2 and 3, and an override of no blocks names none", held)
	}
	if whole.Validate(30) != nil || whole.Validate(20) == nil || (BlockOverride{Start: 0.6, Stop: 0.4}).Validate(30) == nil {
		t.Fatal("a valid override was refused, or one naming a missing block or a reversed span admitted")
	}
}
