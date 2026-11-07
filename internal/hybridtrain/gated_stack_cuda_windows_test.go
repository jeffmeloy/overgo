//go:build windows

package hybridtrain

import (
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/testcheck"
)

// TestAGatedStackTrainsWithItsGate trains a small stack whose attention layer carries Qwen3.5's output gate through the
// hybrid-stack route's layer-streamed lane, its optimizer on the card, and through the host oracle: the lane's losses
// follow the host's step by step, the gate is among the weights that move, and the lane moves it as the host does. The
// device-layer prototype, whose layers carry no gate, refuses the stack rather than train it without one.
func TestAGatedStackTrainsWithItsGate(t *testing.T) {
	// Serial: holds the card.
	cudatest.Require(t)
	const relTol, gateTol = 5e-3, 1e-3
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	cfg := smallHybrid()
	ocfg := optimizer.Config{BaseLearningRate: testLearningRate, Momentum: testMomentum, Schedule: optimizer.ScheduleConstant}
	host := testcheck.Must(buildModel(cfg, testSeed, true))(t)
	dev := testcheck.Must(buildModel(cfg, testSeed, true))(t)
	attention := slices.Index(cfg.Types, FullAttention)
	if attention < 0 || host.Weights[attention].Attn.Gate == nil {
		t.Fatalf("the stack holds no gated attention layer: %v", cfg.Types)
	}
	// Each matrix's gradient packs into the slab range the matrix was bound to, the gate's after Wo: a gradient packed
	// elsewhere moves the wrong weights in the host oracle and the lane alike.
	w := host.Weights[attention]
	fill := func(values []float32, marker float32) []float32 {
		filled := make([]float32, len(values))
		for index := range filled {
			filled[index] = marker
		}
		return filled
	}
	markers := map[string]float32{"mlp.gate": 1, "mlp.up": 2, "mlp.down": 3, "attn.q": 4, "attn.k": 5, "attn.v": 6, "attn.o": 7, "attn.gate": 8}
	grads := hostmath.HybridDecoderLayerGrads{
		DInputNorm: fill(w.InputNorm, 0), DPostNorm: fill(w.PostNorm, 0),
		DMLP: hostmath.QwenMLPWeights{Gate: fill(w.MLP.Gate, 1), Up: fill(w.MLP.Up, 2), Down: fill(w.MLP.Down, 3)},
		DAttn: hostmath.AttentionMixWeights{Wq: fill(w.Attn.Wq, 4), Wk: fill(w.Attn.Wk, 5), Wv: fill(w.Attn.Wv, 6), Wo: fill(w.Attn.Wo, 7),
			Gate: fill(w.Attn.Gate, 8), QNorm: fill(w.Attn.QNorm, 0), KNorm: fill(w.Attn.KNorm, 0)},
	}
	packed := make([]float32, len(host.matW))
	packLayerGradients(grads, packed, make([]float32, len(host.vecW)))
	for _, matrix := range host.mats {
		part, inLayer := strings.CutPrefix(matrix.name, name(attention, ""))
		if !inLayer {
			continue
		}
		for _, value := range packed[matrix.off : matrix.off+matrix.size] {
			if value != markers[part] {
				t.Fatalf("matrix %s's slab range holds a gradient marked %v; want its own, %v", matrix.name, value, markers[part])
			}
		}
	}
	start := slices.Clone(host.Weights[attention].Attn.Gate)
	hostLosses := testcheck.Must(host.TrainHost(testSteps, ocfg, TrainingOptions{}))(t)
	devLosses := testcheck.Must(dev.TrainHostMasterStreamed(worker, testSteps, ocfg, TrainingOptions{}))(t)
	prototype := testcheck.Must(buildModel(cfg, testSeed, true))(t)
	if _, _, err := prototype.TrainDeviceResident(worker, testSteps, ocfg, TrainingOptions{}); err == nil {
		t.Error("the device-layer prototype trained a gated stack its layers carry no gate for")
	}
	worst := 0.0
	for step := range hostLosses {
		worst = max(worst, math.Abs(hostLosses[step]-devLosses[step])/(math.Abs(hostLosses[step])+1e-9))
	}
	hostGate, devGate := host.Weights[attention].Attn.Gate, dev.Weights[attention].Attn.Gate
	var moved, apart, scale float64
	for index, value := range hostGate {
		moved = max(moved, math.Abs(float64(value-start[index])))
		apart = max(apart, math.Abs(float64(devGate[index]-value)))
		scale = max(scale, math.Abs(float64(value)))
	}
	t.Logf("losses %.6f -> %.6f on the host, %.6f -> %.6f through the lane, at most %.3g apart; the gate moved %.3g, the lane's %.3g from the host's",
		hostLosses[0], hostLosses[len(hostLosses)-1], devLosses[0], devLosses[len(devLosses)-1], worst, moved, apart)
	if !(hostLosses[len(hostLosses)-1] < hostLosses[0]) || !(devLosses[len(devLosses)-1] < devLosses[0]) || worst > relTol {
		t.Errorf("the lane's losses %v stand from the host's %v by %.3g, or neither descends", devLosses, hostLosses, worst)
	}
	if moved == 0 || apart > gateTol*scale {
		t.Errorf("the gate moved %.3g and the lane's stands %.3g from the host's of scale %.3g; want it moved alike", moved, apart, scale)
	}
}
