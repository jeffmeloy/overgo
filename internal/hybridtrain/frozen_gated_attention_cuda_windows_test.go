//go:build windows

package hybridtrain

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheFrozenGatedAttentionLayerIsTheHostLayer holds the card's frozen gated full-attention decoder layer to
// hostmath's: on random weights with the output gate, an input and an output gradient, its output and its gradient
// with respect to the input stand within float32 rounding of the host layer's, at a small geometry, two query heads
// on one key head of 4 turned on their first 2 values, and at one with the 9B's heads, 256 wide and turned on their
// first 64, four query heads on one key head.
func TestTheFrozenGatedAttentionLayerIsTheHostLayer(t *testing.T) {
	// Serial: holds the card.
	cudatest.Require(t)
	// tolerance bounds the largest difference from the host, which accumulates in float64, over the largest host value:
	// the card's float32 sums of up to 1024 terms and its fast-math sigmoid, exponent and rotary angles round to under a
	// part in a million here (5.2e-7 at most), and the gate dropped from the forward, the gate's logits' gradient or the
	// rotation's transpose dropped from the backward moves it by more than a part in twenty. theta is Qwen3.5's rotary
	// base.
	const tolerance, eps, theta = 1e-5, 1e-6, 1e7
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	session := testcheck.Must(devicemath.NewResidentOpsSession(worker))(t)
	defer session.Close()
	for _, geometry := range []struct {
		name                                                         string
		tokens, hidden, inter, heads, kvHeads, headDim, rotaryValues int
	}{
		{"small", 6, 8, 16, 2, 1, 4, 2},
		{"9B heads", 32, 256, 512, 4, 1, 256, 64},
	} {
		t.Run(geometry.name, func(t *testing.T) {
			tokens, hidden, inter, width := geometry.tokens, geometry.hidden, geometry.inter, geometry.headDim
			queryDim, keyDim := geometry.heads*width, geometry.kvHeads*width
			rng := rand.New(rand.NewSource(int64(tokens)))
			// A projection's values scale with the root of its inputs, so every activation and gate logit stays near one.
			project := func(rows, columns int) []float32 {
				return testutil.NormalVector(rng, rows*columns, 1/math.Sqrt(float64(columns)))
			}
			scales := func(count int) []float32 { return testutil.NormalVectorAbout(rng, count, 1, 0.1) }
			w := hostmath.HybridLayerWeights{
				InputNorm: scales(hidden), PostNorm: scales(hidden),
				Attn: hostmath.AttentionMixWeights{
					Wq: project(queryDim, hidden), Wk: project(keyDim, hidden), Wv: project(keyDim, hidden), Wo: project(hidden, queryDim),
					QNorm: scales(width), KNorm: scales(width), Gate: project(queryDim, hidden),
				},
				MLP: hostmath.QwenMLPWeights{Gate: project(inter, hidden), Up: project(inter, hidden), Down: project(hidden, inter)},
			}
			d := hostmath.HybridLayerDims{
				Tokens: tokens, Hidden: hidden, Inter: inter, Eps: eps,
				Attn: hostmath.AttentionMixDims{
					Tokens: tokens, Hidden: hidden, Heads: geometry.heads, KVHeads: geometry.kvHeads, HeadDim: width,
					RopeDim: geometry.rotaryValues, RopeTheta: theta, Eps: eps,
				},
			}
			x := testutil.NormalVector(rng, tokens*hidden, 1)
			dOut := testutil.NormalVector(rng, tokens*hidden, 1)
			want, cache := hostmath.HybridDecoderLayerForward(x, w, d, nil)
			wantDX := hostmath.HybridDecoderLayerBackward(x, w, d, nil, dOut, cache).DX

			layer := testcheck.Must(session.OpenFrozenGatedAttentionLayer(w, d))(t)
			defer layer.Close()
			input := testcheck.Must(devicemath.AllocResidentF32(worker, len(x), x))(t)
			gradient := testcheck.Must(devicemath.AllocResidentF32(worker, len(dOut), dOut))(t)
			defer devicemath.FreeResident(worker, input, gradient)
			out := testcheck.Must(layer.Forward(input))(t)
			dx := testcheck.Must(layer.Backward(gradient))(t)
			got, gotDX := make([]float32, len(want)), make([]float32, len(wantDX))
			testcheck.NoError(t, devicemath.ReadResident(worker, out, devicemath.ResidentSlice{Data: got}))
			testcheck.NoError(t, devicemath.ReadResident(worker, dx, devicemath.ResidentSlice{Data: gotDX}))
			for _, pair := range []struct {
				name      string
				got, want []float32
			}{{"output", got, want}, {"input gradient", gotDX, wantDX}} {
				var largest, difference float64
				for index, value := range pair.want {
					largest = max(largest, math.Abs(float64(value)))
					difference = max(difference, math.Abs(float64(pair.got[index])-float64(value)))
				}
				t.Logf("%s: largest difference %.3g of the largest host value %.3g", pair.name, difference/largest, largest)
				if !(difference <= tolerance*largest) {
					t.Errorf("%s: the card stands %.3g of the largest host value from the host, over %g", pair.name, difference/largest, tolerance)
				}
			}
		})
	}
}
