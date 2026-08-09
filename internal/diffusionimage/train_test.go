package diffusionimage

import (
	"math"
	"math/rand"
	"testing"
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
	if err := OTLinearFlowPathInto(xT, target, fx.X1, fx.X0, fx.T, fx.B, n, fx.SigmaMin); err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "x_t", xT, fx.XT, tolTight)
	requireWithin(t, "target", target, fx.Target, tolTight)
	grad := make([]float32, len(fx.Pred))
	loss, err := ScaledMSELossGradInto(grad, fx.Pred, target, fx.LossScale)
	if err != nil {
		t.Fatal(err)
	}
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
	if err := OTLinearFlowPathInto(x, target, x1, make([]float32, len(x1)), times, batch, len(x1)/batch, sigmaMin); err != nil {
		t.Fatal(err)
	}
	return x, target
}

// TestTinyModelTrainDescends: SGD on the tiny fixture model over the fixed
// OT-flow path descends clearly (reference screen: 12 steps to below 0.85x;
// SGD here, so the horizon is longer — optimizer choice recorded in train.go).
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
	losses := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		loss, err := m.TrainStepSGD(x, target, fx.B, fx.H, fx.W, lr)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("non-finite loss %v at step %d", loss, step)
		}
		losses = append(losses, loss)
	}
	first, last := losses[0], losses[len(losses)-1]
	t.Logf("tiny OT-flow SGD loss %.6f -> %.6f over %d steps (lr %g)", first, last, steps, lr)
	if !(last < first*maxFrac) {
		t.Fatalf("loss did not clearly descend: %.6f -> %.6f", first, last)
	}
}

// TestRealCheckpointTrainStepDescends: real-artifact train step — SGD on a
// fixed synthetic batch (vendor data range [-1,1]) decreases the OT-flow
// loss. Fresh model instance: training mutates weights in place.
func TestRealCheckpointTrainStepDescends(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the real artifact; skipped in -short")
	}
	dir := artifactDir(t)
	m, err := Load(dir)
	if err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", dir, err)
	}
	rng := rand.New(rand.NewSource(7))
	x1 := make([]float32, 1*m.Cfg.InChannels*32*32)
	for i := range x1 {
		x1[i] = 2*rng.Float32() - 1
	}
	const (
		steps = 3
		lr    = 1e-4
	)
	x, target := fixedFlowPath(t, x1, 1, 0.001)
	losses := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		loss, err := m.TrainStepSGD(x, target, 1, 32, 32, lr)
		if err != nil {
			t.Fatal(err)
		}
		losses = append(losses, loss)
	}
	t.Logf("real-artifact OT-flow SGD losses %v (lr %g)", losses, lr)
	first, last := losses[0], losses[len(losses)-1]
	if math.IsNaN(last) || math.IsInf(last, 0) {
		t.Fatalf("non-finite final loss %v", last)
	}
	if !(last < first) {
		t.Fatalf("loss did not decrease: %.6f -> %.6f", first, last)
	}
}
