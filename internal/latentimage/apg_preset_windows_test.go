//go:build windows

package latentimage

import (
	"context"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestAPGIsAGuiderPreset holds MOSS's apg preset on the card (guidance-ops/apg-presets): APG replaces the combine of a
// guided step and runs no pass the authors' recipe does not, so a request naming apg runs two passes a step, as the
// reference does, and draws another latent than it.
func TestAPGIsAGuiderPreset(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	const steps = 4
	draw := func(preset string) (passes int, latent []float32) {
		t.Helper()
		request := Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: steps, Seed: 42, Preset: preset}
		g := testcheck.Must(LoadGenerator(t.Context(), root, profile, request))(t)
		defer func() { testcheck.NoError(t, g.Close(context.WithoutCancel(t.Context()))) }()
		g.observe = func(name string, values []float32) error {
			switch {
			case strings.HasPrefix(name, "prediction."):
				passes++
			case strings.HasPrefix(name, "latents."):
				latent = slices.Clone(values)
			}
			return nil
		}
		session := testcheck.Must(g.prepare(t.Context(), request))(t)
		testcheck.Must(g.integrate(t.Context(), session))(t)
		return passes, latent
	}
	_, reference := draw("")
	passes, adapted := draw("apg")
	t.Logf("apg ran %d predictions over %d steps", passes, steps)
	if passes != 2*steps {
		t.Errorf("apg ran %d predictions over %d steps, want the reference's %d", passes, steps, 2*steps)
	}
	if slices.Equal(adapted, reference) {
		t.Error("apg reached the authors' latent")
	}
}
