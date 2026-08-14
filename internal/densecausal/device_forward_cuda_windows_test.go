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

// TestDeviceLayerForwardCachedMatchesHost gates the device layer forward against
// host layerForwardCached on the seeded tiny model: the advanced residual stream
// and every cached intermediate must agree within fp32 tolerance.
func TestDeviceLayerForwardCachedMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	m := tinyMuonModel(t)
	const seq = 5
	d := m.Dims
	rng := rand.New(rand.NewSource(37))
	x := make([]float32, seq*d.Hidden)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.5)
	}
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	invF32 := make([]float32, len(invFreq))
	for i := range invFreq {
		invF32[i] = float32(invFreq[i])
	}
	l0, err := m.layerWeights(0)
	if err != nil {
		t.Fatal(err)
	}

	xHost := append([]float32(nil), x...)
	cacheHost := m.layerForwardCached(xHost, l0, invFreq, seq)
	xDev := append([]float32(nil), x...)
	cacheDev, err := m.deviceLayerForwardCached(worker, xDev, l0, invF32, seq)
	if err != nil {
		t.Fatal(err)
	}

	maxAbs := func(a, b []float32) float64 {
		var mx float64
		for i := range a {
			if v := math.Abs(float64(a[i]) - float64(b[i])); v > mx {
				mx = v
			}
		}
		return mx
	}
	const tolerance = 2e-3
	if v := maxAbs(xHost, xDev); v > tolerance {
		t.Fatalf("advanced x device vs host %.3e > %.1e", v, tolerance)
	} else {
		t.Logf("x: %.3e", v)
	}
	for _, c := range []struct {
		name string
		a, b []float32
	}{
		{"xn", cacheHost.xn, cacheDev.xn},
		{"qScaled", cacheHost.tr.qScaled, cacheDev.tr.qScaled},
		{"kRoped", cacheHost.tr.kRoped, cacheDev.tr.kRoped},
		{"v", cacheHost.tr.v, cacheDev.tr.v},
		{"attnCore", cacheHost.tr.attnCore, cacheDev.tr.attnCore},
		{"h2", cacheHost.h2, cacheDev.h2},
		{"hn", cacheHost.hn, cacheDev.hn},
		{"gate", cacheHost.gate, cacheDev.gate},
		{"up", cacheHost.up, cacheDev.up},
		{"a", cacheHost.a, cacheDev.a},
		{"hMLP", cacheHost.hMLP, cacheDev.hMLP},
	} {
		if v := maxAbs(c.a, c.b); v > tolerance {
			t.Fatalf("cache %s device vs host %.3e > %.1e", c.name, v, tolerance)
		} else {
			t.Logf("%s: %.3e", c.name, v)
		}
	}
}
