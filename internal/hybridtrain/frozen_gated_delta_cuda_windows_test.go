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

// TestTheFrozenGatedDeltaLayerIsTheHostLayer holds the card's frozen gated-delta decoder layer to hostmath's: on random
// weights, an input and an output gradient, its output and its gradient with respect to the input stand within float32
// rounding of the host layer's, at a small geometry and at one with the 9B's 128-wide heads, two value heads a key head
// and a convolution of four.
func TestTheFrozenGatedDeltaLayerIsTheHostLayer(t *testing.T) {
	// Serial: holds the card.
	cudatest.Require(t)
	// tolerance bounds the largest difference from the host, which accumulates in float64, over the largest host value:
	// the card's float32 sums of up to 512 terms and its float32 scan over 32 tokens round to under a part in a million
	// here (7e-7 at most), and a lost term of either gradient moves it by more than a part in a hundred.
	const tolerance, eps = 1e-5, 1e-6
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	session := testcheck.Must(devicemath.NewResidentOpsSession(worker))(t)
	defer session.Close()
	for _, geometry := range []struct {
		name                                                        string
		tokens, hidden, inter, keyHeads, valueHeads, headDim, convK int
	}{
		{"small", 6, 8, 16, 2, 2, 4, 3},
		{"wide heads", 32, 256, 512, 2, 4, 128, 4},
	} {
		t.Run(geometry.name, func(t *testing.T) {
			hidden, inter, heads, width := geometry.hidden, geometry.inter, geometry.valueHeads, geometry.headDim
			keyDim, valDim := geometry.keyHeads*width, heads*width
			rng := rand.New(rand.NewSource(int64(geometry.tokens)))
			// A projection's values scale with the root of its inputs, so every activation stays near one.
			project := func(rows, columns int) []float32 {
				return testutil.NormalVector(rng, rows*columns, 1/math.Sqrt(float64(columns)))
			}
			scales := func(count int) []float32 { return testutil.NormalVectorAbout(rng, count, 1, 0.1) }
			conv := func(channels int) []float32 { return testutil.NormalVector(rng, channels*geometry.convK, 0.5) }
			// A is negative, as Qwen3.5's -exp(A_log) is, so each token's gate decays the state.
			decay := testutil.NormalVector(rng, heads, 1)
			for index, value := range decay {
				decay[index] = -float32(math.Abs(float64(value)))
			}
			w := hostmath.HybridLayerWeights{
				InputNorm: scales(hidden), PostNorm: scales(hidden), IsLinear: true,
				GDN: hostmath.GatedDeltaMixWeights{
					Wq: project(keyDim, hidden), Wk: project(keyDim, hidden), Wv: project(valDim, hidden),
					ConvQ: conv(keyDim), ConvK: conv(keyDim), ConvV: conv(valDim),
					Wbeta: project(heads, hidden), Walpha: project(heads, hidden), TimeStep: testutil.NormalVector(rng, heads, 0.5), A: decay,
					Wz: project(valDim, hidden), Norm: scales(width), Wout: project(hidden, valDim),
				},
				MLP: hostmath.QwenMLPWeights{Gate: project(inter, hidden), Up: project(inter, hidden), Down: project(hidden, inter)},
			}
			d := hostmath.HybridLayerDims{
				Tokens: geometry.tokens, Hidden: hidden, Inter: inter, Eps: eps,
				GDN: hostmath.GatedDeltaMixDims{
					Tokens: geometry.tokens, Hidden: hidden, KeyHeads: geometry.keyHeads, ValueHeads: heads,
					HeadDim: width, ConvK: geometry.convK, OutDim: hidden, Eps: eps,
				},
			}
			x := testutil.NormalVector(rng, geometry.tokens*hidden, 1)
			dOut := testutil.NormalVector(rng, geometry.tokens*hidden, 1)
			zero := make([]float32, heads*width*width)
			want, cache := hostmath.HybridDecoderLayerForward(x, w, d, zero)
			wantDX := hostmath.HybridDecoderLayerBackward(x, w, d, zero, dOut, cache).DX

			layer := testcheck.Must(session.OpenFrozenGatedDeltaLayer(w, d))(t)
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
