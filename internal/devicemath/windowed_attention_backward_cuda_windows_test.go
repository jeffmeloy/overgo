//go:build windows

package devicemath

import (
	"fmt"

	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

// TestWindowedCausalAttentionBackwardDeviceMatchesHost proves the device
// sliding-window backward reproduces hostmath.WindowedCausalAttentionBackward
// across windows spanning the full causal prefix (window<=0), sub-seq bands, an
// exactly-seq band, and an over-seq band (>= full prefix). All three grads must
// be bit-adjacent (~1e-4 f32-GEMM class). window<=0 must match the full-causal
// golden exactly, guarding no regression of the pre-existing full-causal path.
func TestWindowedCausalAttentionBackwardDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const seq, nh, nkv, hd = 7, 4, 2, 8
	rng := rand.New(rand.NewSource(1234))
	q := randSlice(rng, seq*nh*hd)
	k := randSlice(rng, seq*nkv*hd)
	v := randSlice(rng, seq*nkv*hd)
	dOut := randSlice(rng, seq*nh*hd)

	maxDiff := func(got, want []float32) float64 {
		var m float64
		for i := range got {
			if d := math.Abs(float64(got[i]) - float64(want[i])); d > m {
				m = d
			}
		}
		return m
	}

	const tolerance = 1e-4
	for _, window := range []int{-1, 0, 1, 3, seq, seq + 5} {
		hdq := make([]float32, seq*nh*hd)
		hdk := make([]float32, seq*nkv*hd)
		hdv := make([]float32, seq*nkv*hd)
		hostmath.WindowedCausalAttentionBackward(hdq, hdk, hdv, q, k, v, dOut, seq, nh, nkv, hd, window)

		dQ, dK, dV, err := WindowedCausalAttentionBackwardDevice(worker, q, k, v, dOut, seq, nh, nkv, hd, window)
		if err != nil {
			t.Fatalf("window=%d: %v", window, err)
		}
		dqD, dkD, dvD := maxDiff(dQ, hdq), maxDiff(dK, hdk), maxDiff(dV, hdv)
		t.Logf("window=%2d: dQ %.3e dK %.3e dV %.3e", window, dqD, dkD, dvD)
		if dqD > tolerance || dkD > tolerance || dvD > tolerance {
			t.Fatalf("window=%d: dQ %.3e / dK %.3e / dV %.3e > %.1e", window, dqD, dkD, dvD, tolerance)
		}
	}
}

// WindowedCausalAttentionBackwardDevice computes dQ/dK/dV of sliding-window
// causal multi-head (GQA) attention on the device -- the VJP counterpart of
// hostmath.WindowedCausalAttention and the device analogue of
// hostmath.WindowedCausalAttentionBackward. It materializes the windowed causal
// softmax band (window<=0 => full causal prefix, bit-identical to the pre-existing
// full-causal device path) and runs the verified device MultiHeadAttentionBackward.
// The window enters only through the zeroed p band, so no attention kernel changes
// and masked keys contribute no gradient. Score scale is 1 (caller folds
// 1/sqrt(headDim) into q, matching the forward). Q is [seq, heads*headDim]; K/V
// are [seq, kvHeads*headDim]; dOut is [seq, heads*headDim].
func WindowedCausalAttentionBackwardDevice(worker *device.Worker, q, k, v, dOut []float32, seq, heads, kvHeads, headDim, window int) (dQ, dK, dV []float32, err error) {
	if seq <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads%kvHeads != 0 ||
		len(q) != seq*heads*headDim || len(k) != seq*kvHeads*headDim ||
		len(v) != seq*kvHeads*headDim || len(dOut) != seq*heads*headDim {
		return nil, nil, nil, fmt.Errorf("WindowedCausalAttentionBackwardDevice: shape mismatch (seq=%d heads=%d kvHeads=%d headDim=%d)", seq, heads, kvHeads, headDim)
	}
	p := windowedCausalSoftmaxGQA(q, k, seq, heads, kvHeads, headDim, window)
	return MultiHeadAttentionBackward(worker, q, k, v, p, dOut, seq, heads, kvHeads, headDim, 1.0)
}
