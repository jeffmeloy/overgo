//go:build windows

package sensenovarecipe

import (
	"encoding/json"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/routedlm"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// TestSenseNovaRunsCFGZeroAndAGuidanceIntervalAsPresets holds SenseNova's guidance presets to the branches they imply on
// the card (guidance-ops/sensenova-presets): its request named no preset and its body ran both branches every step
// (2026-11-11), so CFG-Zero* and a guidance interval, Wan's and MOSS's presets, could not run on it. A request naming
// cfg-zero-star runs no branch on its zeroed first step and both on every other; one naming guidance-interval runs the
// conditioned branch alone on a step whose noise level lies outside the interval; each reaches another latent than the
// authors' recipe, which runs both branches every step as it did. The requests are read from their words and the passes
// from the stats' words, so the check reads them the same way before and after the change.
func TestSenseNovaRunsCFGZeroAndAGuidanceIntervalAsPresets(t *testing.T) {
	cudatest.Require(t)
	var oracle generationLeadershipOracle
	testcheck.NoError(t, jsonfile.Decode(senseNovaGenerationGold, &oracle))
	const steps = 6
	_, _, generator := newProductionGenerationFixture(t)
	type drawn struct {
		passes int
		latent []float32
	}
	draw := func(preset string) drawn {
		t.Helper()
		words := testcheck.Must(json.Marshal(map[string]any{
			"prompt": oracle.Request.Prompt, "width": oracle.Request.Width, "height": oracle.Request.Height, "steps": steps,
			"seed": oracle.Request.Seed, "cfg_scale": oracle.Request.CFGScale, "timestep_shift": oracle.Request.TimestepShift,
			"preset": preset,
		}))(t)
		var request GenerationRequest
		testcheck.NoError(t, json.Unmarshal(words, &request))
		plan := testcheck.Must(generator.prepare(t.Context(), request))(t)
		features := testcheck.Must(generator.integrate(t.Context(), plan))(t)
		var stats struct{ Passes int }
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(json.Marshal(generator.Stats()))(t), &stats))
		return drawn{passes: stats.Passes, latent: slices.Clone(features.patches)}
	}
	var declared map[string]struct {
		Guidance sampling.Guidance `json:"guidance"`
	}
	testcheck.NoError(t, jsonfile.Decode("presets.json", &declared))
	interval := declared["guidance-interval"].Guidance.Interval
	schedule := testcheck.Must(routedlm.ShiftedFlowTimeSchedule(steps, oracle.Request.TimestepShift))(t)
	guided := 0
	for _, time := range schedule[:steps] {
		if noise := 1 - time; interval.Low <= noise && noise <= interval.High {
			guided++
		}
	}
	if guided == 0 || guided == steps || oracle.Request.CFGScale == 1 {
		t.Fatalf("the interval [%g, %g] holds %d of the schedule's %d noise levels at guidance %g: no step to skip, or none to guide", interval.Low, interval.High, guided, steps, oracle.Request.CFGScale)
	}
	authors := draw("")
	for _, preset := range []struct {
		name   string
		passes int
	}{
		{"", 2 * steps},
		{"cfg-zero-star", 2 * (steps - 1)},
		{"guidance-interval", steps + guided},
	} {
		run := authors
		if preset.name != "" {
			run = draw(preset.name)
		}
		t.Logf("%q ran %d branches over %d steps", preset.name, run.passes, steps)
		if run.passes != preset.passes {
			t.Errorf("%q ran %d branches over %d steps, want %d", preset.name, run.passes, steps, preset.passes)
		}
		if preset.name != "" && slices.Equal(run.latent, authors.latent) {
			t.Errorf("%q reached the authors' latent", preset.name)
		}
	}
}
