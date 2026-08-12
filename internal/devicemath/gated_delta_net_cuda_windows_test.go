//go:build windows

package devicemath

import (
	"context"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

// TestGDNForwardMatchesKernel pins hostmath.GatedDeltaNetForward against the
// gated_delta_net_f32 CUDA kernel it ports -- the forward the GDN VJP will
// differentiate. Output and final state must match within fp32 tolerance.
func TestGDNForwardMatchesKernel(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const size, qHeads, kHeads, heads, tokens, seqs, gateWidth = 8, 2, 2, 2, 4, 1, 1
	rng := rand.New(rand.NewSource(19))
	rs := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * 0.3)
		}
		return s
	}
	query := rs(size * qHeads * tokens * seqs)
	key := rs(size * kHeads * tokens * seqs)
	value := rs(size * heads * tokens * seqs)
	gate := rs(gateWidth * heads * tokens * seqs)
	beta := rs(heads * tokens * seqs)
	inputState := rs(heads * seqs * size * size)

	wantOut, wantState := hostmath.GatedDeltaNetForward(
		query, key, value, gate, beta, inputState,
		size, qHeads, kHeads, heads, tokens, seqs, gateWidth, false)

	attn := size * heads * tokens * seqs
	stateN := heads * seqs * size * size
	out := make([]float32, attn+stateN)
	err = worker.Do(context.Background(), func(state *device.State) error {
		lib := state.Driver
		module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return err
		}
		defer lib.ModuleUnload(module)
		fn, err := lib.ModuleFunction(module, "gated_delta_net_f32")
		if err != nil {
			return err
		}
		var ptrs []driver.DevicePtr
		up := func(d []float32) (driver.DevicePtr, error) {
			p, err := lib.MemAlloc(uint64(len(d)) * 4)
			if err != nil {
				return 0, err
			}
			ptrs = append(ptrs, p)
			return p, lib.MemcpyHtoD(p, driver.Bytes(d))
		}
		defer func() {
			for _, p := range ptrs {
				lib.MemFree(p)
			}
		}()
		qP, _ := up(query)
		kP, _ := up(key)
		vP, _ := up(value)
		gP, _ := up(gate)
		bP, _ := up(beta)
		sP, _ := up(inputState)
		oP, err := up(out)
		if err != nil {
			return err
		}
		sz, qh, kh, hd := uint32(size), uint32(qHeads), uint32(kHeads), uint32(heads)
		tk, sq, gw, ri := uint32(tokens), uint32(seqs), uint32(gateWidth), uint32(0)
		args := []unsafe.Pointer{
			unsafe.Pointer(&qP), unsafe.Pointer(&kP), unsafe.Pointer(&vP), unsafe.Pointer(&gP),
			unsafe.Pointer(&bP), unsafe.Pointer(&sP), unsafe.Pointer(&oP),
			unsafe.Pointer(&sz), unsafe.Pointer(&qh), unsafe.Pointer(&kh), unsafe.Pointer(&hd),
			unsafe.Pointer(&tk), unsafe.Pointer(&sq), unsafe.Pointer(&gw), unsafe.Pointer(&ri),
		}
		blocks := uint32(heads * seqs)
		if err := lib.LaunchKernel(fn,
			driver.Dim3{X: blocks, Y: 1, Z: 1}, driver.Dim3{X: 256, Y: 1, Z: 1},
			0, state.Stream, args); err != nil {
			return err
		}
		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(out), oP)
	})
	if err != nil {
		t.Fatal(err)
	}

	maxAbs := func(a, b []float32) float64 {
		var m float64
		for i := range a {
			if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
				m = d
			}
		}
		return m
	}
	outDiff := maxAbs(wantOut, out[:attn])
	stateDiff := maxAbs(wantState, out[attn:])
	t.Logf("GDN forward: output |dev-host| %.3e, final-state %.3e", outDiff, stateDiff)
	const tol = 1e-4
	if outDiff > tol || stateDiff > tol {
		t.Fatalf("GDN forward host vs kernel: out %.3e state %.3e > %.1e", outDiff, stateDiff, tol)
	}
}

// TestGatedDeltaNetBackwardMatchesKernel pins hostmath.GatedDeltaNetBackward (the
// FD-verified host BPTT golden) against the gated_delta_net_backward_f32 CUDA
// kernel via GatedDeltaNetBackwardDevice. All six gradients must match within the
// forward's fp32 tolerance class. Covers both gate widths and GQA grouping so the
// grouped q/k atomic accumulation and per-column gate path are exercised.
func TestGatedDeltaNetBackwardMatchesKernel(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	maxAbs := func(a, b []float32) float64 {
		var m float64
		for i := range a {
			if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
				m = d
			}
		}
		return m
	}

	cases := []struct {
		name                                                 string
		size, qHeads, kHeads, heads, tokens, seqs, gateWidth int
		repeatInterleave                                     bool
	}{
		{"scalar_gate", 8, 2, 2, 2, 4, 1, 1, false},
		{"vector_gate", 8, 2, 2, 2, 4, 2, 8, false},
		{"gqa_interleave", 6, 2, 1, 4, 3, 2, 1, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(1009 + tc.size + tc.heads)))
			rs := func(n int) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(rng.NormFloat64() * 0.3)
				}
				return s
			}
			query := rs(tc.size * tc.qHeads * tc.tokens * tc.seqs)
			key := rs(tc.size * tc.kHeads * tc.tokens * tc.seqs)
			value := rs(tc.size * tc.heads * tc.tokens * tc.seqs)
			gate := rs(tc.gateWidth * tc.heads * tc.tokens * tc.seqs)
			beta := rs(tc.heads * tc.tokens * tc.seqs)
			inputState := rs(tc.heads * tc.seqs * tc.size * tc.size)
			dOutput := rs(tc.size * tc.heads * tc.tokens * tc.seqs)

			wantDQ, wantDK, wantDV, wantDG, wantDB, wantDS := hostmath.GatedDeltaNetBackward(
				query, key, value, gate, beta, inputState, dOutput,
				tc.size, tc.qHeads, tc.kHeads, tc.heads, tc.tokens, tc.seqs, tc.gateWidth,
				tc.repeatInterleave)

			gotDQ, gotDK, gotDV, gotDG, gotDB, gotDS, err := GatedDeltaNetBackwardDevice(
				worker, query, key, value, gate, beta, inputState, dOutput,
				tc.size, tc.qHeads, tc.kHeads, tc.heads, tc.tokens, tc.seqs, tc.gateWidth,
				tc.repeatInterleave)
			if err != nil {
				t.Fatal(err)
			}

			dqD := maxAbs(wantDQ, gotDQ)
			dkD := maxAbs(wantDK, gotDK)
			dvD := maxAbs(wantDV, gotDV)
			dgD := maxAbs(wantDG, gotDG)
			dbD := maxAbs(wantDB, gotDB)
			dsD := maxAbs(wantDS, gotDS)
			t.Logf("GDN backward: dQ %.3e dK %.3e dV %.3e dGate %.3e dBeta %.3e dState %.3e",
				dqD, dkD, dvD, dgD, dbD, dsD)
			const tol = 1e-4
			if dqD > tol || dkD > tol || dvD > tol || dgD > tol || dbD > tol || dsD > tol {
				t.Fatalf("GDN backward host vs kernel exceeds %.1e: dQ %.3e dK %.3e dV %.3e dGate %.3e dBeta %.3e dState %.3e",
					tol, dqD, dkD, dvD, dgD, dbD, dsD)
			}
		})
	}
}
