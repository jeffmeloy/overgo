package latentimage

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestMossLayerSkipIsTheGenericBlockOverride holds MOSS's negative-branch layer skip to the generic block override
// (sampling.BlockOverride) a guider carries: MOSS's profile names block 22, skipped on the negative branch at every step
// of its schedule, which is the override over the whole schedule the rotary sampler reads, and so its default steps skip
// block 22 at each of them as the special case did.
func TestMossLayerSkipIsTheGenericBlockOverride(t *testing.T) {
	t.Parallel()
	moss := testcheck.Must(lookupCatalogProfile("MOSSImagePipeline"))(t)
	if moss.Rotary == nil {
		t.Fatal("MOSS's profile has no rotary policy")
	}
	override := moss.Rotary.negativeOverride()
	if !slices.Equal(override.Blocks, []int{22}) || override.Start != 0 || override.Stop != 1 {
		t.Fatalf("MOSS's negative branch overrides %+v, want block 22 over the whole schedule", override)
	}
	spec := rotarySamplingSpec{guidance: moss.Rotary.Guidance, override: override, schedule: FlowSchedule{Steps: moss.DefaultSteps}}
	for step := range moss.DefaultSteps {
		if !slices.Equal(spec.override.At(step, spec.schedule.Steps), []int{22}) {
			t.Fatalf("step %d of %d skips %v on the negative branch, want block 22", step, moss.DefaultSteps, spec.override.At(step, spec.schedule.Steps))
		}
	}
}
