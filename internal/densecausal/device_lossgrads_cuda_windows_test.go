//go:build windows

package densecausal

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

// TestDeviceLossAndGradsMatchesHost gates the full-model device backward against
// host LossAndGrads on the tiny model: loss, logits and every parameter gradient
// must agree within the promoted BF16-operand, FP32-accumulation floor.
func TestDeviceLossAndGradsMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	m := tinyMuonModel(t)
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}

	lossHost, logitsHost, gHost, err := m.LossAndGrads(tokens)
	if err != nil {
		t.Fatal(err)
	}
	lossDev, logitsDev, gDev, err := m.deviceLossAndGrads(worker, tokens)
	if err != nil {
		t.Fatal(err)
	}

	if d := math.Abs(lossHost - lossDev); d > 5e-5 {
		t.Fatalf("loss host %.6f device %.6f (|d|=%.3e)", lossHost, lossDev, d)
	}
	if v := testutil.MaxAbsDiff(logitsHost, logitsDev); v > 1e-3 {
		t.Fatalf("logits host vs device %.3e", v)
	}
	if len(gHost) != len(gDev) {
		t.Fatalf("grad tensor count device %d != host %d", len(gDev), len(gHost))
	}
	var worst float64
	for name, want := range gHost {
		got, ok := gDev[name]
		if !ok {
			t.Fatalf("device missing grad %q", name)
		}
		if v := testutil.MaxAbsDiff(want, got); v > worst {
			worst = v
		}
		if v := testutil.MaxAbsDiff(want, got); v > 2e-3 {
			t.Fatalf("grad %q host vs device %.3e", name, v)
		}
	}
	t.Logf("loss |d| ok; worst weight grad %.3e over %d tensors", worst, len(gHost))
}

func TestDeviceQwen2BiasLossAndGradsMatchHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	golden := readQwen2Golden(t)
	model := modelFromGolden(t, golden)
	wantLoss, wantLogits, wantGrads, err := model.LossAndGrads(golden.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	gotLoss, gotLogits, gotGrads, err := model.deviceLossAndGrads(worker, golden.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	if delta := math.Abs(wantLoss - gotLoss); delta > 5e-5 {
		t.Fatalf("Qwen2 loss delta %.3e", delta)
	}
	if delta := testutil.MaxAbsDiff(wantLogits, gotLogits); delta > 1e-3 {
		t.Fatalf("Qwen2 logits delta %.3e", delta)
	}
	for name, want := range wantGrads {
		got, ok := gotGrads[name]
		if !ok {
			t.Fatalf("Qwen2 device gradient %q absent", name)
		}
		if delta := testutil.MaxAbsDiff(want, got); delta > 2e-3 {
			t.Fatalf("Qwen2 gradient %q delta %.3e", name, delta)
		}
	}
}
