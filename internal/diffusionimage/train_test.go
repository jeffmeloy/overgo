package diffusionimage

import (
	"math"
	"testing"

	"overgo/internal/checkpointtest"
	"overgo/internal/optimizer"
	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

// TestOTFlowObjectiveMatchesTorch: path, target, scaled-MSE loss and
// gradient vs the torch golden (train.py OptimalTransportLinearFlowGenerator
// with the vendor's scale 69).
func TestOTFlowObjectiveMatchesTorch(t *testing.T) {
	fx := loadGolden[struct {
		B, C, H, W          int
		SigmaMin, LossScale float64
		X1, X0, T           []float32
		Pred, XT, Target    []float32
		Loss                float64
		DPred               []float32
	}](t, "ot_flow")
	n := fx.C * fx.H * fx.W
	xT := make([]float32, len(fx.X1))
	target := make([]float32, len(fx.X1))
	testcheck.NoError(t, OTLinearFlowPathInto(xT, target, fx.X1, fx.X0, fx.T, fx.B, n, fx.SigmaMin))
	requireWithin(t, "x_t", xT, fx.XT, tolTight)
	requireWithin(t, "target", target, fx.Target, tolTight)
	grad := make([]float32, len(fx.Pred))
	loss := testcheck.Must(ScaledMSELossGradInto(grad, fx.Pred, target, fx.LossScale))(t)
	t.Logf("ot_flow loss got %.9g want %.9g", loss, fx.Loss)
	if math.Abs(loss-fx.Loss) > tolTight {
		t.Fatalf("loss=%g want=%g", loss, fx.Loss)
	}
	requireWithin(t, "gradient", grad, fx.DPred, tolTight)
}

// fixedFlowPath: the reference train screen's deterministic objective —
// one OT path built once at t=1 with zero noise (x ~= x1, target ~= x1), so
// per-step losses are comparable and descent is measurable.
func fixedFlowPath(t *testing.T, x1 []float32, batch int, sigmaMin float64) (x, target []float32) {
	t.Helper()
	x = make([]float32, len(x1))
	target = make([]float32, len(x1))
	times := make([]float32, batch)
	for i := range times {
		times[i] = 1
	}
	testcheck.NoError(t, OTLinearFlowPathInto(x, target, x1, make([]float32, len(x1)), times, batch, len(x1)/batch, sigmaMin))
	return x, target
}

// TestTinyModelTrainDescends: Muon on a fixed OT-flow path.
func TestTinyModelTrainDescends(t *testing.T) {
	fx := loadGolden[uditTinyFixture](t, "udit_tiny")
	m := compileTiny(t, fx)
	const (
		steps    = 30
		lr       = 0.02
		sigmaMin = 0.001
		maxFrac  = 0.85
	)
	x, target := fixedFlowPath(t, fx.X, fx.B, sigmaMin)
	trainer := testcheck.Must(NewTrainer(m, optimizer.Config{BaseLearningRate: lr, Momentum: 0.95, Schedule: optimizer.ScheduleConstant}))(t)
	if trainer.Program().Objective() != trainingprogram.ObjectiveFlowMatching {
		t.Fatalf("trainer objective = %q", trainer.Program().Objective())
	}
	defer func() {
		testcheck.NoError(t, trainer.Close())
	}()
	losses := make([]float64, 0, steps)
	for step := range steps {
		loss := testcheck.Must(trainer.Step(x, target, fx.B, fx.H, fx.W))(t)
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("non-finite loss %v at step %d", loss, step)
		}
		losses = append(losses, loss)
	}
	first, last := losses[0], losses[len(losses)-1]
	t.Logf("tiny OT-flow Muon loss %.6f -> %.6f over %d steps (lr %g)", first, last, steps, lr)
	if !(last < first*maxFrac) {
		t.Fatalf("loss did not clearly descend: %.6f -> %.6f", first, last)
	}
}

// TestTrainerCheckpointRoundTripIsBitExact steps the tiny model once,
// publishes every tensor and the optimizer state as a checkpoint, and reads
// both back bit for bit.
func TestTrainerCheckpointRoundTripIsBitExact(t *testing.T) {
	fx := loadGolden[uditTinyFixture](t, "udit_tiny")
	m := compileTiny(t, fx)
	config := testcheck.Must(trainingprogram.BuiltinOptimizerPolicy().Config(m.TrainableParameterCount()))(t)
	trainer := testcheck.Must(NewTrainer(m, config))(t)
	defer trainer.Close()
	x, target := fixedFlowPath(t, fx.X, fx.B, 0.001)
	if _, err := trainer.Step(x, target, fx.B, fx.H, fx.W); err != nil {
		t.Fatal(err)
	}
	if !(trainer.GradientL2() > 0) {
		t.Fatalf("step gradient norm %g, want a gradient", trainer.GradientL2())
	}
	checkpointtest.RequireRoundTrip(t, trainer.Program(), trainer, m.raw)
}

// TestRealCheckpointMuonPlanCompiles: every real tensor has Muon geometry.
func TestRealCheckpointMuonPlanCompiles(t *testing.T) {
	requireLongTest(t)
	dir := artifactDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", dir, err)
	}
	trainer := testcheck.Must(NewTrainer(m, optimizer.Config{BaseLearningRate: 1e-4, Momentum: 0.95, Schedule: optimizer.ScheduleConstant}))(t)
	defer func() {
		testcheck.NoError(t, trainer.Close())
	}()
	if trainer.pack.ParameterCount() == 0 {
		t.Fatal("real-artifact Muon plan is empty")
	}
}
