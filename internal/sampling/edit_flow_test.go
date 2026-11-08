package sampling

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestCompileEditFlowSigmas(t *testing.T) {
	t.Parallel()
	got := testcheck.Must(CompileEditFlowSigmas(EditFlowConfig{InferenceSteps: 4, TrainTimesteps: 4, Shift: 1, SigmaMax: 1}, []int64{4, 0}))(t)
	if len(got) != 2 || got[0] != 1 || got[1] != 0 {
		t.Fatalf("sigmas=%v", got)
	}
}

// TestAWarpedEditFlowReadsTheScheduleFromItsEnd holds Self-Forcing's warp_denoising_step: its four steps 1000, 750,
// 500 and 250 of a 1000-step schedule shifted by 5 are the schedule's entries 0, 250, 500 and 750, at sigmas 1,
// 0.9375, 0.8333 and 0.625, which its network reads as timesteps 1000, 937.5, 833.3 and 625; LiveEdit's same list,
// unwarped, runs at 1, 0.75, 0.5006 and 0.2516.
func TestAWarpedEditFlowReadsTheScheduleFromItsEnd(t *testing.T) {
	t.Parallel()
	config := EditFlowConfig{InferenceSteps: 1000, TrainTimesteps: 1000, Shift: 5, SigmaMax: 1, ExtraStep: true, Warp: true}
	got := testcheck.Must(CompileEditFlowSigmas(config, []int64{1000, 750, 500, 250}))(t)
	want := []float32{1, 0.9375, 0.8333333, 0.625}
	for index := range want {
		if difference := got[index] - want[index]; difference > 1e-6 || difference < -1e-6 {
			t.Fatalf("warped sigmas %v, want %v", got, want)
		}
	}
	if _, err := CompileEditFlowSigmas(config, []int64{1001}); err == nil {
		t.Fatal("a warped timestep before the schedule's start was given a sigma")
	}
}
