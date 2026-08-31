package diffusionimage

import (
	"math"
	"testing"
	"time"

	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// TestRealCheckpointTrainingSmoke: bounded observed training on the real
// SimpleDiffusion-TPA-Rope checkpoint — the training mirror of the
// inference validation path. The pair is the committed real-forward image
// on the reference's deterministic OT path (fixedFlowPath), every step is
// observed, and descent is the same pair's loss falling across the steps.
func TestRealCheckpointTrainingSmoke(t *testing.T) {
	requireLongTest(t)
	const (
		steps    = 3
		lr       = 1e-4
		sigmaMin = 0.001
	)
	dir := artifactDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", dir, err)
	}
	golden := loadGolden[struct {
		B, H, W int
		X       []float32
	}](t, "real_forward")
	x, target := fixedFlowPath(t, golden.X, golden.B, sigmaMin)
	trainer, err := NewTrainer(m, optimizer.Config{BaseLearningRate: lr, Momentum: 0.95, Schedule: optimizer.ScheduleConstant})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := trainer.Close(); err != nil {
			t.Fatal(err)
		}
	}()
	if trainer.Program().Objective() != trainingprogram.ObjectiveFlowMatching {
		t.Fatalf("trainer objective = %q", trainer.Program().Objective())
	}
	t.Logf("trainable parameters=%d image=%dx%dx%d pixels=%d", trainer.pack.ParameterCount(), golden.B, golden.H, golden.W, len(golden.X))
	start := time.Now()
	losses := make([]float64, 0, steps)
	for step := range steps {
		stepStart := time.Now()
		loss, err := trainer.Step(x, target, golden.B, golden.H, golden.W)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("non-finite loss %v at step %d", loss, step+1)
		}
		losses = append(losses, loss)
		t.Logf("step %d/%d: loss=%.6f wall=%s", step+1, steps, loss, time.Since(stepStart).Round(time.Millisecond))
	}
	first, last := losses[0], losses[len(losses)-1]
	if !(last < first) {
		t.Fatalf("real training loss did not descend: %.6f -> %.6f over %d steps", first, last, steps)
	}
	t.Logf("real checkpoint training smoke: steps=%d loss %.6f -> %.6f total_wall=%s",
		steps, first, last, time.Since(start).Round(time.Millisecond))
}
