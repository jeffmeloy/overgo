//go:build windows

package devicegrad

import (
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestWindowedCausalAttentionBackwardDeviceMatchesHost proves the device
// sliding-window backward reproduces testutil.WindowedCausalAttentionBackward
// across windows spanning the full causal prefix (window<=0), sub-seq bands, an
// exactly-seq band, and an over-seq band (>= full prefix). All three grads must
// be bit-adjacent (~1e-4 f32-GEMM class). window<=0 must match the full-causal
// golden exactly, guarding no regression of the pre-existing full-causal path.
func TestWindowedCausalAttentionBackwardDeviceMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	testcheck.NoError(t, err)
	defer worker.Close()

	const seq, nh, nkv, hd = 7, 4, 2, 8
	rng := rand.New(rand.NewSource(1234))
	randSlice := func(n int) []float32 { return testutil.NormalVector(rng, n, 1) }
	q := randSlice(seq * nh * hd)
	k := randSlice(seq * nkv * hd)
	v := randSlice(seq * nkv * hd)
	dOut := randSlice(seq * nh * hd)

	maxDiff := testutil.MaxAbsDiff[float32, float32]

	const tolerance = 1e-4
	for _, window := range []int{-1, 0, 1, 3, seq, seq + 5} {
		hdq := make([]float32, seq*nh*hd)
		hdk := make([]float32, seq*nkv*hd)
		hdv := make([]float32, seq*nkv*hd)
		testutil.WindowedCausalAttentionBackward(hdq, hdk, hdv, q, k, v, dOut, seq, nh, nkv, hd, window)

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
