package latentvideo

import (
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// TestTheGuiderTakesCFGZeroAndAGuidanceInterval holds Wan's guidance presets to the passes they imply. A request naming
// cfg-zero-star runs the authors' recipe with one zeroed step and the projected unconditioned output; one naming
// guidance-interval guides only the steps whose noise level lies within it. Over a trajectory, CFG-Zero*'s zeroed step
// runs no network pass and every other step two; a step outside the interval runs the conditioned pass alone. The
// reference preset runs two passes a step, as it did.
func TestTheGuiderTakesCFGZeroAndAGuidanceInterval(t *testing.T) {
	t.Parallel()
	profile := testcheck.Must(declaredProfile())(t)
	zeroStar := testcheck.Must(profile.complete(WanRequest{Prompt: "a fox", Preset: "cfg-zero-star"}))(t)
	interval := testcheck.Must(profile.complete(WanRequest{Prompt: "a fox", Preset: "guidance-interval"}))(t)
	reference := testcheck.Must(profile.complete(WanRequest{Prompt: "a fox"}))(t)
	if zeroStar.Guidance != (sampling.Guidance{ZeroSteps: 1, Projected: true}) || interval.Guidance.Interval == (sampling.NoiseInterval{}) ||
		reference.Guidance != (sampling.Guidance{}) || zeroStar.Steps != reference.Steps || zeroStar.GuideScale != reference.GuideScale {
		t.Fatalf("the presets gave %+v and %+v against the authors' recipe %+v", zeroStar.Guidance, interval.Guidance, reference.Guidance)
	}
	program, request := layoutFixture(1, 2, 4)
	request.GuideScale = 5
	schedule := func() ([]float32, error) {
		_, sigmas, err := sampling.UniPCSchedule(program.Config.Policy.NumTrainTimesteps, request.Steps, request.Shift)
		return sigmas, err
	}
	sigmas := testcheck.Must(schedule())(t)
	band := sampling.NoiseInterval{Low: float64(sigmas[request.Steps-1]), High: float64(sigmas[1])}
	guided := 0
	for _, sigma := range sigmas[:request.Steps] {
		if band.Low <= float64(sigma) && float64(sigma) <= band.High {
			guided++
		}
	}
	for _, guidance := range []struct {
		name     string
		guidance sampling.Guidance
		passes   int
	}{
		{"the reference", sampling.Guidance{}, 2 * request.Steps},
		{"CFG-Zero*", sampling.Guidance{ZeroSteps: 1, Projected: true}, 2 * (request.Steps - 1)},
		{"a guidance interval", sampling.Guidance{Interval: band}, request.Steps + guided},
	} {
		backend := &countingBackend{}
		request.Guidance = guidance.guidance
		testcheck.Must(program.DenoiseWithBackend(backend, request))(t)
		t.Logf("%s ran %d network passes over %d steps", guidance.name, backend.passes, request.Steps)
		if backend.passes != guidance.passes {
			t.Errorf("%s ran %d network passes over %d steps, want %d", guidance.name, backend.passes, request.Steps, guidance.passes)
		}
	}
}
