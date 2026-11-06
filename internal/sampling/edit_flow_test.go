package sampling

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestCompileEditFlowSigmas(t *testing.T) {
	got := testcheck.Must(CompileEditFlowSigmas(EditFlowConfig{InferenceSteps: 4, TrainTimesteps: 4, Shift: 1, SigmaMax: 1}, []int64{4, 0}))(t)
	if len(got) != 2 || got[0] != 1 || got[1] != 0 {
		t.Fatalf("sigmas=%v", got)
	}
}
