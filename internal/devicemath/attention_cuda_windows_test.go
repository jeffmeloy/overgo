//go:build windows

package devicemath

import (
	"fmt"

	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// hostAttention computes single-head causal attention in float64, returning the
// causal-masked softmax p (fp32, as the device backward consumes it) and the
// output.
func hostAttention(q, k, v []float32, seq, hd int, scale float64) (p []float32, out []float64) {
	p = make([]float32, seq*seq)
	out = make([]float64, seq*hd)
	row := make([]float64, seq)
	for i := range seq {
		mx := math.Inf(-1)
		for j := 0; j <= i; j++ {
			var dot float64
			for h := range hd {
				dot += float64(q[i*hd+h]) * float64(k[j*hd+h])
			}
			row[j] = dot * scale
			if row[j] > mx {
				mx = row[j]
			}
		}
		var sum float64
		for j := 0; j <= i; j++ {
			row[j] = math.Exp(row[j] - mx)
			sum += row[j]
		}
		for j := 0; j <= i; j++ {
			pij := row[j] / sum
			p[i*seq+j] = float32(pij)
			for h := range hd {
				out[i*hd+h] += pij * float64(v[j*hd+h])
			}
		}
	}
	return p, out
}

func TestAttentionCoreBackwardResidentMatchesShared(t *testing.T) {
	cudatest.Require(t)
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	const seq, headDim = 8, 6
	scale := 1 / math.Sqrt(float64(headDim))
	rng := rand.New(rand.NewSource(47))
	q, k, v, dOut := randSlice(rng, seq*headDim), randSlice(rng, seq*headDim), randSlice(rng, seq*headDim), randSlice(rng, seq*headDim)
	probability, _ := hostAttention(q, k, v, seq, headDim, scale)
	want := testcheck.Must(AttentionCoreBackward(worker, q, k, v, probability, dOut, seq, headDim, scale))(t)
	var pointers []driver.DevicePtr
	allocate := residentAllocation(t, worker, &pointers)
	defer func() {
		if err := FreeResident(worker, pointers...); err != nil {
			t.Error(err)
		}
	}()
	qPtr, kPtr, vPtr := allocate(len(q), q), allocate(len(k), k), allocate(len(v), v)
	probabilityPtr, dOutPtr := allocate(len(probability), probability), allocate(len(dOut), dOut)
	dQPtr, dKPtr, dVPtr := allocate(len(q), nil), allocate(len(k), nil), allocate(len(v), nil)
	dScoresPtr := allocate(len(probability), nil)
	if err := AttentionCoreBackwardResident(
		worker, qPtr, kPtr, vPtr, probabilityPtr, dOutPtr,
		dQPtr, dKPtr, dVPtr, dScoresPtr, seq, headDim, scale,
	); err != nil {
		t.Fatal(err)
	}
	gotQ, gotK, gotV, gotScores := make([]float32, len(q)), make([]float32, len(k)), make([]float32, len(v)), make([]float32, len(probability))
	for _, read := range []struct {
		pointer driver.DevicePtr
		data    []float32
	}{{dQPtr, gotQ}, {dKPtr, gotK}, {dVPtr, gotV}, {dScoresPtr, gotScores}} {
		testcheck.NoError(t, ReadResident(worker, read.pointer, ResidentSlice{Data: read.data}))
	}
	qDelta, kDelta := testutil.MaxAbsDiff(gotQ, want.DQ), testutil.MaxAbsDiff(gotK, want.DK)
	vDelta, scoreDelta := testutil.MaxAbsDiff(gotV, want.DV), testutil.MaxAbsDiff(gotScores, want.DScores)
	t.Logf("resident/shared attention VJP dQ=%.3e dK=%.3e dV=%.3e dS=%.3e", qDelta, kDelta, vDelta, scoreDelta)
	if max(qDelta, kDelta, vDelta, scoreDelta) > 2e-7 {
		t.Fatalf("resident attention VJP differs")
	}
}

// TestAttentionCoreBackwardGradCheck verifies device AttentionCoreBackward
// (dQ/dK/dV) against float64 central finite differences of L = sum(dOut ⊙ out).
func TestAttentionCoreBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()

	const seq, hd = 6, 4
	scale := 1 / math.Sqrt(float64(hd))
	rng := rand.New(rand.NewSource(9))
	q := randSlice(rng, seq*hd)
	k := randSlice(rng, seq*hd)
	v := randSlice(rng, seq*hd)
	dOut := randSlice(rng, seq*hd)

	p, _ := hostAttention(q, k, v, seq, hd, scale)
	grads := testcheck.Must(AttentionCoreBackward(worker, q, k, v, p, dOut, seq, hd, scale))(t)

	loss := func() float64 {
		_, out := hostAttention(q, k, v, seq, hd, scale)
		var l float64
		for i := range out {
			l += float64(dOut[i]) * out[i]
		}
		return l
	}
	const eps = 1e-3
	gradCheck := func(name string, param, analytic []float32) float64 {
		return finiteDifference(t, name, param, analytic, eps, loss)
	}
	const tolerance = 3e-3
	worstScore := 0.0
	for query := range seq {
		dp := make([]float64, seq)
		var expectation float64
		for key := 0; key <= query; key++ {
			for channel := range hd {
				dp[key] += float64(dOut[query*hd+channel]) * float64(v[key*hd+channel])
			}
			expectation += float64(p[query*seq+key]) * dp[key]
		}
		for key := range seq {
			want := float64(p[query*seq+key]) * (dp[key] - expectation)
			worstScore = max(worstScore, math.Abs(float64(grads.DScores[query*seq+key])-want))
		}
	}
	t.Logf("dScores: max |device-host| %.3e", worstScore)
	worst := gradCheck("dQ", q, grads.DQ)
	worst = max(worst, gradCheck("dK", k, grads.DK))
	worst = max(worst, gradCheck("dV", v, grads.DV))
	worst = max(worst, worstScore)
	if worst > tolerance {
		t.Fatalf("worst grad-check %.3e > %.1e", worst, tolerance)
	}
}

// AttentionCoreBackwardResident writes one attention VJP to device buffers.
func AttentionCoreBackwardResident(
	worker *device.Worker,
	q, k, v, probability, dOut driver.DevicePtr,
	dQ, dK, dV, dScores driver.DevicePtr,
	seq, headDim int,
	scale float64,
) error {
	if worker == nil || q == 0 || k == 0 || v == 0 || probability == 0 || dOut == 0 || dQ == 0 || dK == 0 || dV == 0 || dScores == 0 || seq <= 0 || headDim <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return fmt.Errorf("AttentionCoreBackwardResident: invalid buffer, geometry, or scale")
	}
	return WithResidentOps(worker, func(ops *ResidentOps) error {
		return ops.AttentionCoreBackward(q, k, v, probability, dOut, dQ, dK, dV, dScores, seq, headDim, scale)
	})
}
