//go:build windows

package latentimage

import (
	"errors"
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

// TestADenoiseCutsTheScheduleAsComfyUIDoes holds a request's denoise (moss-ultra-sections/img2img) to ComfyUI's
// KSampler: below 1 a schedule, named or the profile's own, is drawn for int(steps / denoise) steps and its last
// steps + 1 sigmas kept with their timesteps and deltas; above 0.9999 the whole schedule is sampled; a denoise outside
// (0, 1], or one beside a list of the request's own sigmas, is refused; a request may leave its prompt empty, as the
// MOSS Ultra workflow's sections do, where a flow profile refuses one; and only a rotary profile takes a picture or a
// denoise.
func TestADenoiseCutsTheScheduleAsComfyUIDoes(t *testing.T) {
	profile := Profile{Sampling: samplingPolicy{DefaultSteps: 50, TrainTimesteps: 1000}, Rotary: &rotaryPolicy{Shift: 5, TrainingShift: 3}}
	named := &sampling.Schedule{Name: sampling.ScheduleBongTangent}
	cut := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Steps: 10, Schedule: named, Denoise: 0.6}))(t)
	whole := testcheck.Must(sampling.BongTangent(16))(t)
	if cut.Steps != 10 || len(cut.Sigmas) != 11 || len(cut.Timesteps) != 10 || len(cut.Deltas) != 10 {
		t.Fatalf("10 steps at denoise 0.6 kept %d steps of %d sigmas", cut.Steps, len(cut.Sigmas))
	}
	for index, sigma := range cut.Sigmas {
		if sigma != float64(whole[index+6]) {
			t.Fatalf("10 steps at denoise 0.6 kept sigmas %v; want the last 11 of bong_tangent at int(10 / 0.6) = 16 steps, %v", cut.Sigmas, whole[6:])
		}
	}
	if cut.Timesteps[0] != float64(whole[6]*1000) || cut.Deltas[0] != float64(whole[7]-whole[6]) {
		t.Errorf("the cut schedule's timesteps %v and deltas %v are not its sigmas'", cut.Timesteps, cut.Deltas)
	}
	own := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Steps: 8, Denoise: 0.5}))(t)
	drawn := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Steps: 16}))(t)
	if !slices.Equal(own.Sigmas, drawn.Sigmas[8:]) {
		t.Errorf("the profile's own schedule at 8 steps and denoise 0.5 kept %v; want the last 9 of its 16, %v", own.Sigmas, drawn.Sigmas[8:])
	}
	full := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Steps: 10, Schedule: named, Denoise: 0.99995}))(t)
	ten := testcheck.Must(profile.rotarySchedule(t.Context(), Request{Steps: 10, Schedule: named}))(t)
	if !slices.Equal(full.Sigmas, ten.Sigmas) {
		t.Errorf("a denoise above 0.9999 sampled %v; want the whole schedule, %v", full.Sigmas, ten.Sigmas)
	}
	for _, request := range []Request{
		{Prompt: "a lighthouse", Width: 1024, Height: 576, Denoise: 1.5},
		{Prompt: "a lighthouse", Width: 1024, Height: 576, Denoise: -0.2},
		{Prompt: "a lighthouse", Width: 1024, Height: 576, Denoise: 0.5, Schedule: &sampling.Schedule{Sigmas: []float32{1, 0.5, 0}}},
	} {
		if ValidateRequest(request) == nil {
			t.Errorf("the request %+v was admitted; want it refused", request)
		}
	}
	testcheck.NoError(t, ValidateRequest(Request{Width: 1024, Height: 576, Image: "cGljdHVyZQ==", Denoise: 0.6}))
	testcheck.NoError(t, ValidateRequest(Request{Width: 1024, Height: 576}))
	if err := (&Generator{}).Reset(t.Context(), Request{Width: 1024, Height: 576}); !errors.Is(err, errEmptyPrompt) {
		t.Errorf("a flow session took a request of no prompt with %v; want it refused", err)
	}
	if !(Request{Image: "cGljdHVyZQ=="}).rotaryOnly() || !(Request{Denoise: 0.6}).rotaryOnly() {
		t.Error("a request with a picture or a denoise is run by a profile that is not rotary")
	}
}
