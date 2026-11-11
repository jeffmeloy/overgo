//go:build windows

package latentimage

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// TestMOSSRunsCFGZeroAndAGuidanceIntervalAsPresets holds MOSS's guidance presets to the passes they imply on the card
// (guidance-ops/moss-presets): its sampling compiled one update a step and ran both passes on every step with no guider
// (2026-11-11), so CFG-Zero* and a guidance interval, Wan's presets since guidance-ops/guider, could not run on it. A
// request naming cfg-zero-star runs no pass on its zeroed first step and both on every other; one naming
// guidance-interval runs the conditioned pass alone on a step whose noise level lies outside the interval; each draws
// another latent than the authors' recipe, which runs both passes every step as it did.
func TestMOSSRunsCFGZeroAndAGuidanceIntervalAsPresets(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	const steps = 6
	reference := Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: steps, Seed: 42}
	// draw integrates a request and counts the predictions its sampling makes, keeping the noise levels of its schedule
	// and the latent it reaches.
	type drawn struct {
		passes int
		sigmas []float64
		latent []float32
	}
	draw := func(request Request) drawn {
		t.Helper()
		g := testcheck.Must(LoadGenerator(t.Context(), root, profile, request))(t)
		defer func() { testcheck.NoError(t, g.Close(context.WithoutCancel(t.Context()))) }()
		var result drawn
		g.observe = func(name string, values []float32) error {
			switch {
			case strings.HasPrefix(name, "prediction."):
				result.passes++
			case strings.HasPrefix(name, "latents."):
				result.latent = slices.Clone(values)
			}
			return nil
		}
		session := testcheck.Must(g.prepare(t.Context(), request))(t)
		testcheck.Must(g.integrate(t.Context(), session))(t)
		result.sigmas = g.schedule.Sigmas
		return result
	}
	authors := draw(reference)
	// The band is read from the catalog's words, so the check reads it the same way before and after the change.
	type declared struct {
		Pipeline string
		Presets  map[string]struct {
			Guidance sampling.Guidance `json:"guidance"`
		}
	}
	var catalog []declared
	testcheck.NoError(t, json.Unmarshal(profileCatalogJSON, &catalog))
	moss := slices.IndexFunc(catalog, func(profile declared) bool { return profile.Pipeline == "MOSSImagePipeline" })
	if moss < 0 {
		t.Fatal("the catalog holds no MOSS profile")
	}
	interval := catalog[moss].Presets["guidance-interval"].Guidance.Interval
	guided := 0
	for _, sigma := range authors.sigmas[:steps] {
		if interval.Low <= sigma && sigma <= interval.High {
			guided++
		}
	}
	if guided == 0 || guided == steps {
		t.Fatalf("the interval [%g, %g] holds %d of the schedule's %d noise levels %v: no step to skip, or none to guide", interval.Low, interval.High, guided, steps, authors.sigmas)
	}
	for _, preset := range []struct {
		name   string
		passes int
	}{
		{"", 2 * steps},
		{"cfg-zero-star", 2 * (steps - 1)},
		{"guidance-interval", steps + guided},
	} {
		request := reference
		request.Preset = preset.name
		run := authors
		if preset.name != "" {
			run = draw(request)
		}
		t.Logf("%q ran %d predictions over %d steps", preset.name, run.passes, steps)
		if run.passes != preset.passes {
			t.Errorf("%q ran %d predictions over %d steps, want %d", preset.name, run.passes, steps, preset.passes)
		}
		if preset.name != "" && slices.Equal(run.latent, authors.latent) {
			t.Errorf("%q reached the authors' latent", preset.name)
		}
	}
}
