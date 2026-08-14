//go:build windows

package densecausal

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

// TestDeviceLayerBackwardMatchesHost gates the full densecausal-exact device
// layer backward against the host layerBackward on the seeded tiny model: the
// input gradient dx and every weight gradient must agree within fp32 tolerance.
func TestDeviceLayerBackwardMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	m := tinyMuonModel(t) // Hidden 16, 2 heads x hd 8, 2 kv heads, inter 32, no attn bias
	const seq = 5
	d := m.Dims
	rng := rand.New(rand.NewSource(31))
	x := make([]float32, seq*d.Hidden)
	dOut := make([]float32, seq*d.Hidden)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.5)
		dOut[i] = float32(rng.NormFloat64())
	}
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)

	gHost := Grads{}
	dxHost, err := m.layerBackward(0, x, dOut, invFreq, seq, gHost)
	if err != nil {
		t.Fatal(err)
	}
	gDev := Grads{}
	l0, err := m.layerWeights(0)
	if err != nil {
		t.Fatal(err)
	}
	xForCache := append([]float32(nil), x...)
	cache := m.layerForwardCached(xForCache, l0, invFreq, seq)
	dxDev, err := m.deviceLayerBackward(worker, 0, x, dOut, cache, invFreq, seq, gDev)
	if err != nil {
		t.Fatal(err)
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
	const tolerance = 2e-3
	if v := maxAbs(dxHost, dxDev); v > tolerance {
		t.Fatalf("dx device vs host %.3e > %.1e", v, tolerance)
	} else {
		t.Logf("dx: %.3e", v)
	}
	if len(gHost) != len(gDev) {
		t.Fatalf("grad tensor count device %d != host %d", len(gDev), len(gHost))
	}
	var worst float64
	var worstName string
	for name, want := range gHost {
		got, ok := gDev[name]
		if !ok {
			t.Fatalf("device missing grad %q", name)
		}
		if len(got) != len(want) {
			t.Fatalf("grad %q len device %d != host %d", name, len(got), len(want))
		}
		v := maxAbs(want, got)
		if v > worst {
			worst = v
			worstName = name
		}
		if v > tolerance {
			t.Fatalf("grad %q device vs host %.3e > %.1e", name, v, tolerance)
		}
	}
	t.Logf("worst weight grad: %.3e (%s) over %d tensors", worst, worstName, len(gHost))
}
