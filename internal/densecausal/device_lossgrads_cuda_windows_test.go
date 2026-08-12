//go:build windows

package densecausal

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestDeviceLossAndGradsMatchesHost gates the full-model device backward against
// host LossAndGrads on the tiny model: loss, logits and every parameter gradient
// must agree within fp32 tolerance.
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

	if d := math.Abs(lossHost - lossDev); d > 1e-5 {
		t.Fatalf("loss host %.6f device %.6f (|d|=%.3e)", lossHost, lossDev, d)
	}
	maxAbs := func(a, b []float32) float64 {
		var m float64
		for i := range a {
			if v := math.Abs(float64(a[i]) - float64(b[i])); v > m {
				m = v
			}
		}
		return m
	}
	if v := maxAbs(logitsHost, logitsDev); v > 1e-4 {
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
		if v := maxAbs(want, got); v > worst {
			worst = v
		}
		if v := maxAbs(want, got); v > 2e-3 {
			t.Fatalf("grad %q host vs device %.3e", name, v)
		}
	}
	t.Logf("loss |d| ok; worst weight grad %.3e over %d tensors", worst, len(gHost))
}
