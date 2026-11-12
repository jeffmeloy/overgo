package sampling

import (
	"slices"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// comfySchedules is ComfyUI's KSampler's own sigmas under the MOSS Ultra workflow's schedulers, MOSS's model sampling
// and RES4LYF loaded (scripts/python_golden_references/comfy_schedules_golden.py): a schedule ComfyUI could not draw
// holds its error instead.
type comfySchedules struct {
	ModelSampling struct {
		Shift      float64 `json:"shift"`
		Multiplier float64 `json:"multiplier"`
		SigmaMin   float64 `json:"sigma_min"`
		SigmaMax   float64 `json:"sigma_max"`
	} `json:"model_sampling"`
	Schedules []struct {
		Scheduler string    `json:"scheduler"`
		Steps     int       `json:"steps"`
		Denoise   float64   `json:"denoise"`
		Error     string    `json:"error"`
		Sigmas    []float64 `json:"sigmas"`
	} `json:"schedules"`
}

// flowTableLevels is the levels of MOSS's discrete flow sampling in ComfyUI.
const flowTableLevels = 1000

// TestBongTangentEqualsTheRecordedSigmas holds the named schedules (moss-ultra-sections/schedules) to ComfyUI's own:
// bong_tangent, written from its formula, gives exactly the sigmas ComfyUI's KSampler drew with RES4LYF at the Ultra
// workflow's 8 and 50 steps, at its image-to-image denoise of 0.72, and at steps on either side of its slope's scale;
// beta57 over MOSS's flow levels gives the ones drawn at 8 and 50 steps and at denoise 0.72; and where ComfyUI failed,
// bong_tangent at one step, the schedule is refused.
func TestBongTangentEqualsTheRecordedSigmas(t *testing.T) {
	var recorded comfySchedules
	testcheck.NoError(t, jsonfile.Decode("testdata/comfy_schedules.json", &recorded))
	table := testcheck.Must(ShiftedFlowTable(flowTableLevels, recorded.ModelSampling.Shift))(t)
	if float64(table[0]) != recorded.ModelSampling.SigmaMin || float64(table[len(table)-1]) != recorded.ModelSampling.SigmaMax {
		t.Errorf("flow levels run %g to %g, ComfyUI's %g to %g", table[0], table[len(table)-1], recorded.ModelSampling.SigmaMin, recorded.ModelSampling.SigmaMax)
	}
	drawn := map[string]bool{}
	for _, schedule := range recorded.Schedules {
		sigmas, err := NamedSchedule(schedule.Scheduler, schedule.Steps, schedule.Denoise, table)
		if schedule.Error != "" {
			if err == nil {
				t.Errorf("%s at %d steps, denoise %g: drawn %v where ComfyUI failed (%s)", schedule.Scheduler, schedule.Steps, schedule.Denoise, sigmas, schedule.Error)
			}
			continue
		}
		want := make([]float32, len(schedule.Sigmas))
		for i, sigma := range schedule.Sigmas {
			want[i] = float32(sigma)
		}
		if err != nil || !slices.Equal(sigmas, want) {
			t.Errorf("%s at %d steps, denoise %g:\n got %v (%v)\nwant %v", schedule.Scheduler, schedule.Steps, schedule.Denoise, sigmas, err, want)
		}
		if schedule.Scheduler == ScheduleBongTangent && (schedule.Steps == 8 || schedule.Steps == 50) {
			drawn[schedule.Scheduler+map[bool]string{true: " denoised", false: ""}[schedule.Denoise < 1]] = true
		}
	}
	if !drawn[ScheduleBongTangent] || !drawn[ScheduleBongTangent+" denoised"] {
		t.Errorf("the recording holds no bong_tangent at the workflow's steps, whole and denoised: %v", drawn)
	}
}

// TestARequestsOwnSigmasAreHeldToAFlowSchedule holds a request's own sigma list: descending within [0, 1].
func TestARequestsOwnSigmasAreHeldToAFlowSchedule(t *testing.T) {
	for _, sigmas := range [][]float32{{1, 0.5, 0}, {0.9, 0.2}} {
		testcheck.NoError(t, ValidateFlowSigmas(sigmas))
	}
	for _, sigmas := range [][]float32{{1}, {1.5, 0}, {0.5, 0.5, 0}, {0.2, 0.4, 0}, {1, -0.1}} {
		if ValidateFlowSigmas(sigmas) == nil {
			t.Errorf("%v was taken as a flow schedule", sigmas)
		}
	}
	if _, err := NamedSchedule("karras", 8, 1, nil); err == nil {
		t.Error("an unknown schedule was drawn")
	}
}
