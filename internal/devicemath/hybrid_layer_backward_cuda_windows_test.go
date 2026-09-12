//go:build windows

package devicemath

import (
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/testutil"
)

// TestHybridDecoderLayerBackwardDeviceMatchesHost proves the device VJP
// (HybridDecoderLayerBackwardDevice) reproduces hostmath.HybridDecoderLayerBackward
// for BOTH hybrid mix variants on random weights/input: full_attention with a
// PARTIAL-rope config (RopeDim < HeadDim, exercising the serving partial-rope
// path) and linear_attention (the GDN mix, exercising the device L2Norm/ShortConv/
// GDN backward composition). Every grad group is compared to the host golden.
func TestHybridDecoderLayerBackwardDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

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
		dOut := sc(T * H)

		_, c := hostmath.HybridDecoderLayerForward(x, w, d, nil)
		want := hostmath.HybridDecoderLayerBackward(x, w, d, nil, dOut, c)
		got, err := HybridDecoderLayerBackwardDevice(worker, x, w, d, nil, dOut)
		if err != nil {
			t.Fatal(err)
		}
		verifyHybridCachedBackward(t, worker, x, w, d, nil, dOut, got)

		check := func(name string, a, b []float32) {
			diff := testutil.MaxAbsDiff(a, b)
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
		// small scale keeps the GDN recurrence well-conditioned for FD-class parity.
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
			Wz: sc(valDim * H), Norm: sc(hd), Wout: sc(H * valDim),
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
		dOut := sc(T * H)

		_, c := hostmath.HybridDecoderLayerForward(x, w, d, state)
		want := hostmath.HybridDecoderLayerBackward(x, w, d, state, dOut, c)
		got, err := HybridDecoderLayerBackwardDevice(worker, x, w, d, state, dOut)
		if err != nil {
			t.Fatal(err)
		}
		verifyHybridCachedBackward(t, worker, x, w, d, state, dOut, got)
		if !got.IsLinear {
			t.Fatal("device grads not flagged IsLinear for the GDN mix")
		}

		check := func(name string, a, b []float32) {
			diff := testutil.MaxAbsDiff(a, b)
			if diff > tol {
				t.Errorf("%-12s max|device-host| %.3e > %.1e", name, diff, tol)
			} else {
				t.Logf("%-12s max|device-host| %.3e", name, diff)
			}
		}
		// layer-level grad groups
		check("dX", got.DX, want.DX)
		check("dInputNorm", got.DInputNorm, want.DInputNorm)
		check("dPostNorm", got.DPostNorm, want.DPostNorm)
		check("dMLP.Gate", got.DMLP.Gate, want.DMLP.Gate)
		check("dMLP.Up", got.DMLP.Up, want.DMLP.Up)
		check("dMLP.Down", got.DMLP.Down, want.DMLP.Down)
		check("dState", got.DState, want.DState)
		// every GDN-mix grad group
		check("dWq", got.DGDN.DWq, want.DGDN.DWq)
		check("dWk", got.DGDN.DWk, want.DGDN.DWk)
		check("dWv", got.DGDN.DWv, want.DGDN.DWv)
		check("dConvQ", got.DGDN.DConvQ, want.DGDN.DConvQ)
		check("dConvK", got.DGDN.DConvK, want.DGDN.DConvK)
		check("dConvV", got.DGDN.DConvV, want.DGDN.DConvV)
		check("dConvBiasQ", got.DGDN.DConvBiasQ, want.DGDN.DConvBiasQ)
		check("dConvBiasK", got.DGDN.DConvBiasK, want.DGDN.DConvBiasK)
		check("dConvBiasV", got.DGDN.DConvBiasV, want.DGDN.DConvBiasV)
		check("dWbeta", got.DGDN.DWbeta, want.DGDN.DWbeta)
		check("dWalpha", got.DGDN.DWalpha, want.DGDN.DWalpha)
		check("dTimeStep", got.DGDN.DTimeStep, want.DGDN.DTimeStep)
		check("dA", got.DGDN.DA, want.DGDN.DA)
		check("dWz", got.DGDN.DWz, want.DGDN.DWz)
		check("dNorm", got.DGDN.DNorm, want.DGDN.DNorm)
		check("dWout", got.DGDN.DWout, want.DGDN.DWout)
	})
}
