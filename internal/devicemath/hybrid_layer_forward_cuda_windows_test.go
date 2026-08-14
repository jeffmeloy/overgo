//go:build windows

package devicemath

import (
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

// TestHybridDecoderLayerForwardDeviceMatchesHost proves the device forward
// (HybridDecoderLayerForwardDevice) reproduces hostmath.HybridDecoderLayerForward's
// layer output for BOTH hybrid mix variants on random weights/input:
// full_attention with a PARTIAL-rope config (RopeDim < HeadDim, exercising the
// serving partial-rope path) and linear_attention (the GDN mix). The device
// output is compared to the host golden `out` to the kernel-parity class.
func TestHybridDecoderLayerForwardDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	// Parity class of the sibling backward test (same configs, same fp32 attention
	// softmax path). The logged per-variant magnitudes show the true tightness:
	// the GDN mix lands ~2e-6; full_attention ~1.5e-4 (dominated by the fp32
	// causal-softmax accumulation, exactly as HybridDecoderLayerBackwardDevice).
	const tol = 1e-3

	t.Run("full_attention", func(t *testing.T) {
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

		want, _ := hostmath.HybridDecoderLayerForward(x, w, d, nil)
		got, cache, err := HybridDecoderLayerForwardDevice(worker, x, w, d, nil)
		if err != nil {
			t.Fatal(err)
		}
		if cache.IsLinear {
			t.Fatal("device cache flagged IsLinear for the attention mix")
		}
		diff := maxAbsDiff(got, want)
		if diff > tol {
			t.Errorf("out          max|device-host| %.3e > %.1e", diff, tol)
		} else {
			t.Logf("out          max|device-host| %.3e", diff)
		}
	})

	t.Run("linear_attention", func(t *testing.T) {
		const (
			T     = 7
			H     = 32
			inter = 48
			hk    = 2
			hv    = 2
			hd    = 8
			convK = 3
			eps   = 1e-6
		)
		keyDim, valDim := hk*hd, hv*hd
		rng := rand.New(rand.NewSource(31))
		// small scale keeps the GDN recurrence well-conditioned for parity.
		sc := func(n int) []float32 {
			s := make([]float32, n)
			for i := range s {
				s[i] = float32(rng.NormFloat64() * 0.4)
			}
			return s
		}

		gdn := hostmath.GatedDeltaMixWeights{
			Wq: sc(keyDim * H), Wk: sc(keyDim * H), Wv: sc(valDim * H),
			ConvQ: sc(keyDim * convK), ConvK: sc(keyDim * convK), ConvV: sc(valDim * convK),
			ConvBiasQ: sc(keyDim), ConvBiasK: sc(keyDim), ConvBiasV: sc(valDim),
			Wbeta: sc(hv * H), Walpha: sc(hv * H), TimeStep: sc(hv), A: sc(hv),
			Wz: sc(valDim * H), Norm: sc(valDim), Wout: sc(H * valDim),
		}
		w := hostmath.HybridLayerWeights{
			InputNorm: sc(H), PostNorm: sc(H), IsLinear: true,
			GDN: gdn,
			MLP: hostmath.QwenMLPWeights{Gate: sc(inter * H), Up: sc(inter * H), Down: sc(H * inter)},
		}
		d := hostmath.HybridLayerDims{Tokens: T, Hidden: H, Inter: inter, Eps: eps,
			GDN: hostmath.GatedDeltaMixDims{Tokens: T, Hidden: H, KeyHeads: hk, ValueHeads: hv, HeadDim: hd, ConvK: convK, OutDim: H, Eps: eps}}
		x := sc(T * H)
		state := sc(hv * hd * hd)

		want, _ := hostmath.HybridDecoderLayerForward(x, w, d, state)
		got, cache, err := HybridDecoderLayerForwardDevice(worker, x, w, d, state)
		if err != nil {
			t.Fatal(err)
		}
		if !cache.IsLinear {
			t.Fatal("device cache not flagged IsLinear for the GDN mix")
		}
		diff := maxAbsDiff(got, want)
		if diff > tol {
			t.Errorf("out          max|device-host| %.3e > %.1e", diff, tol)
		} else {
			t.Logf("out          max|device-host| %.3e", diff)
		}
	})
}
