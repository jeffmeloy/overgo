//go:build windows

package latentimage

import (
	"slices"
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// TestARotaryRequestSamplesItsOwnSchedule holds a request's schedule on MOSS's rotary profile
// (moss-ultra-sections/schedules): bong_tangent is sampled as ComfyUI draws it, at the profile's default steps where
// the request names none and not snapped to the training grid; a list of the request's own sigmas is its steps; and a
// profile that samples its own schedule refuses the request's rather than drawing another.
func TestARotaryRequestSamplesItsOwnSchedule(t *testing.T) {
	profile := Profile{Sampling: samplingPolicy{DefaultSteps: 50, TrainTimesteps: 1000}, Rotary: &rotaryPolicy{Shift: 5, TrainingShift: 3}}
	snapped := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Steps: 8}))(t)
	named := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Steps: 8, Schedule: &sampling.Schedule{Name: sampling.ScheduleBongTangent}}))(t)
	bong := testcheck.Must(sampling.BongTangent(8))(t)
	if named.Steps != 8 || len(named.Sigmas) != 9 || named.Sigmas[1] != float64(bong[1]) || slices.Equal(named.Sigmas, snapped.Sigmas) {
		t.Errorf("bong_tangent at 8 steps sampled %v, want ComfyUI's %v, not the profile's %v", named.Sigmas, bong, snapped.Sigmas)
	}
	if named.Timesteps[1] != float64(bong[1]*1000) || named.Deltas[0] != float64(bong[1]-bong[0]) {
		t.Errorf("bong_tangent's timesteps %v and deltas %v are not its sigmas'", named.Timesteps, named.Deltas)
	}
	whole := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Schedule: &sampling.Schedule{Name: sampling.ScheduleBongTangent}}))(t)
	if whole.Steps != 50 {
		t.Errorf("a named schedule with no steps drew %d, want the profile's 50", whole.Steps)
	}
	own := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Schedule: &sampling.Schedule{Sigmas: []float32{1, 0.6, 0.25, 0}}}))(t)
	if own.Steps != 3 || !slices.Equal(own.Sigmas, []float64{1, float64(float32(0.6)), 0.25, 0}) {
		t.Errorf("the request's own sigmas sampled %d steps over %v", own.Steps, own.Sigmas)
	}
	if _, err := profile.rotarySchedule(t.Context(), Request{Steps: 8, Schedule: &sampling.Schedule{Sigmas: []float32{1, 0.5, 0}}}); err == nil {
		t.Error("two sigmas were taken as 8 steps")
	}
	if _, err := flowSchedule(Request{Steps: 8, NumTrainTimesteps: 1000, Schedule: &sampling.Schedule{Name: sampling.ScheduleBongTangent}}); err != errDrawsItsOwnSchedule {
		t.Errorf("a flow profile drew a request's schedule: %v", err)
	}
}
