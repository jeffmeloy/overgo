//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

// maxAbsDiff reports the max |a-b| over two equal-length slices.
func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
			m = d
		}
	}
	return m
}

// TestHybridDecoderLayerBackwardDeviceMatchesHost proves the device VJP
// (HybridDecoderLayerBackwardDevice) reproduces hostmath.HybridDecoderLayerBackward
// for the full_attention hybrid decoder layer, on random weights/input, for a
// PARTIAL-rope config (RopeDim < HeadDim) -- exercising the serving partial-rope
// path the device must honor. Every grad group is compared to the host golden.
func TestHybridDecoderLayerBackwardDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const (
		T     = 7
		H     = 32
		inter = 48
		heads = 4
		kv    = 2
		hd    = 8
		// PARTIAL rope: rotate only the first 4 of 8 head dims.
		ropeDim = 4
		theta   = 10000.0
		eps     = 1e-6
	)
	rng := rand.New(rand.NewSource(7))
	sc := func(n int) []float32 { return randSlice(rng, n) }

	w := hostmath.HybridLayerWeights{
		InputNorm: sc(H), PostNorm: sc(H), IsLinear: false,
		MLP: hostmath.QwenMLPWeights{Gate: sc(inter * H), Up: sc(inter * H), Down: sc(H * inter)},
		Attn: hostmath.AttentionMixWeights{
			Wq: sc(heads * hd * H), Wk: sc(kv * hd * H), Wv: sc(kv * hd * H), Wo: sc(H * heads * hd),
			QNorm: sc(hd), KNorm: sc(hd),
		},
	}
	d := hostmath.HybridLayerDims{Tokens: T, Hidden: H, Inter: inter, Eps: eps,
		Attn: hostmath.AttentionMixDims{Tokens: T, Hidden: H, Heads: heads, KVHeads: kv, HeadDim: hd, RopeDim: ropeDim, RopeTheta: theta, Eps: eps}}
	x := sc(T * H)
	dOut := sc(T * H)

	_, c := hostmath.HybridDecoderLayerForward(x, w, d, nil)
	want := hostmath.HybridDecoderLayerBackward(x, w, d, nil, dOut, c)

	got, err := HybridDecoderLayerBackwardDevice(worker, x, w, d, nil, dOut)
	if err != nil {
		t.Fatal(err)
	}

	const tol = 1e-3
	check := func(name string, a, b []float32) {
		diff := maxAbsDiff(a, b)
		if diff > tol {
			t.Errorf("%-12s max|device-host| %.3e > %.1e", name, diff, tol)
		} else {
			t.Logf("%-12s max|device-host| %.3e", name, diff)
		}
	}
	check("dX", got.DX, want.DX)
	check("dInputNorm", got.DInputNorm, want.DInputNorm)
	check("dPostNorm", got.DPostNorm, want.DPostNorm)
	check("dMLP.Gate", got.DMLP.Gate, want.DMLP.Gate)
	check("dMLP.Up", got.DMLP.Up, want.DMLP.Up)
	check("dMLP.Down", got.DMLP.Down, want.DMLP.Down)
	check("dWq", got.DAttn.Wq, want.DAttn.Wq)
	check("dWk", got.DAttn.Wk, want.DAttn.Wk)
	check("dWv", got.DAttn.Wv, want.DAttn.Wv)
	check("dWo", got.DAttn.Wo, want.DAttn.Wo)
	check("dQNorm", got.DAttn.QNorm, want.DAttn.QNorm)
	check("dKNorm", got.DAttn.KNorm, want.DAttn.KNorm)
}

// TestHybridDecoderLayerBackwardDeviceRejectsLinear confirms the GDN mix variant
// is refused (milestone 1b: needs L2Norm/ShortConv backward kernels) rather than
// silently returning wrong grads.
func TestHybridDecoderLayerBackwardDeviceRejectsLinear(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const T, H = 2, 8
	w := hostmath.HybridLayerWeights{InputNorm: make([]float32, H), PostNorm: make([]float32, H), IsLinear: true}
	d := hostmath.HybridLayerDims{Tokens: T, Hidden: H, Inter: 8, Eps: 1e-6}
	if _, err := HybridDecoderLayerBackwardDevice(worker, make([]float32, T*H), w, d, nil, make([]float32, T*H)); err == nil {
		t.Fatal("expected linear_attention mix to be rejected on device")
	}
}
