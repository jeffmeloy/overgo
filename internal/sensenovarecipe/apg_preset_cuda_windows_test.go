//go:build windows

package sensenovarecipe

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// TestAPGIsAGuiderPreset holds SenseNova's apg preset on the card (guidance-ops/apg-presets): APG replaces the combine
// of a guided step and runs no branch the authors' recipe does not, so a request naming apg runs both branches every
// step, as the reference does, and reaches another latent than it.
func TestAPGIsAGuiderPreset(t *testing.T) {
	cudatest.Require(t)
	var oracle generationLeadershipOracle
	testcheck.NoError(t, jsonfile.Decode(senseNovaGenerationGold, &oracle))
	const steps = 6
	_, _, generator := newProductionGenerationFixture(t)
	draw := func(preset string) (int, []float32) {
		t.Helper()
		request := GenerationRequest{Prompt: oracle.Request.Prompt, Width: oracle.Request.Width, Height: oracle.Request.Height, Steps: steps,
			Seed: oracle.Request.Seed, CFGScale: oracle.Request.CFGScale, TimestepShift: oracle.Request.TimestepShift, Preset: preset}
		plan := testcheck.Must(generator.prepare(t.Context(), request))(t)
		features := testcheck.Must(generator.integrate(t.Context(), plan))(t)
		return generator.Stats().Passes, slices.Clone(features.patches)
	}
	_, reference := draw("")
	passes, adapted := draw("apg")
	t.Logf("apg ran %d branches over %d steps", passes, steps)
	if passes != 2*steps {
		t.Errorf("apg ran %d branches over %d steps, want the reference's %d", passes, steps, 2*steps)
	}
	if slices.Equal(adapted, reference) {
		t.Error("apg reached the authors' latent")
	}
}
