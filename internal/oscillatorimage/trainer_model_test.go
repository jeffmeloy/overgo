//go:build modeltest

package oscillatorimage

import (
	"math"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestRealArtifactBootstrapTraining verifies objective band and scratch descent.
func TestRealArtifactBootstrapTraining(t *testing.T) {
	cudatest.Require(t)
	testskip.Short(t)
	real, err := Load(artifactDir(t))
	if err != nil {
		t.Skipf("UNAVAILABLE: Un-0 artifact absent: %v", err)
	}
	realTrainer := testcheck.Must(NewTrainer(real, testTrainingConfig(), 202))(t)
	defer realTrainer.Close()
	shipped := realTrainer.Loss(202)
	if shipped < 15.57-3*1.16 || shipped > 15.57+3*1.16 {
		t.Fatalf("real checkpoint loss %.4f outside the reference shipped band 15.57 +/- 3*1.16; objective identity not established", shipped)
	}
	t.Logf("objective identity: real checkpoint loss %.4f inside shipped band; parameters=%d", shipped, realTrainer.ParameterCount())

	const steps = 12
	scratch := NewBootstrapModel(real.Cfg, 17)
	trainer := testcheck.Must(NewTrainer(scratch, testTrainingConfig(), 17))(t)
	defer trainer.Close()
	before := trainer.Loss(202)
	start := time.Now()
	for step := range steps {
		result := testcheck.Must(trainer.Step())(t)
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) || (step == 0 && result.GradientL2 <= 0) {
			t.Fatalf("step %d invalid: %+v", step+1, result)
		}
		t.Logf("step %d/%d: loss=%.6f lr=%.4g grad_l2=%.4g", step+1, steps, result.Loss, result.LearningRate, result.GradientL2)
	}
	after := trainer.Loss(202)
	if math.IsNaN(after) || math.IsInf(after, 0) || !(after < before) {
		t.Fatalf("scratch bootstrap loss did not descend: %.6f -> %.6f", before, after)
	}
	t.Logf("scratch bootstrap training: steps=%d loss %.6f -> %.6f total_wall=%s",
		steps, before, after, time.Since(start).Round(time.Millisecond))
}
