//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
)

// deviceGEMM computes row-major C[m,n] = op(A)·op(B) in fp32 on the GPU, with
// op(A) = [m,k] and op(B) = [k,n]. a and b are host slices of m*k and k*n
// elements (their transpose does not change the element count). A general
// matmul primitive the attention backward composes.
func deviceGEMM(worker *device.Worker, transA, transB bool, m, k, n int, a, b []float32) ([]float32, error) {
	if m <= 0 || k <= 0 || n <= 0 || len(a) != m*k || len(b) != k*n {
		return nil, fmt.Errorf("deviceGEMM: shape mismatch (m=%d k=%d n=%d a=%d b=%d)", m, k, n, len(a), len(b))
	}
	c := make([]float32, m*n)
	err := withCUDABLAS(worker, func(session *cudaBLAS) error {
		aPtr, err := session.upload(a)
		if err != nil {
			return err
		}
		bPtr, err := session.upload(b)
		if err != nil {
			return err
		}
		cPtr, err := session.alloc(len(c))
		if err != nil {
			return err
		}
		if err := session.gemm(transA, transB, m, k, n, aPtr, bPtr, cPtr); err != nil {
			return err
		}
		return session.finish(cudaDownload{c, cPtr})
	})
	if err != nil {
		return nil, err
	}
	return c, nil
}

// AttentionGrads holds the query/key/value gradients of one attention head.
type AttentionGrads struct {
	DQ      []float32 // [seq, hd]
	DK      []float32 // [seq, hd]
	DV      []float32 // [seq, hd]
	DScores []float32 // [seq, seq]
}

// AttentionCoreBackward computes the dQ/dK/dV of single-head scaled dot-product
// attention
//
//	scores = (Q·Kᵀ)*scale ; p = softmax(scores) [causal] ; out = p·V
//
// given the output cotangent dOut[seq,hd] and the saved softmax p[seq,seq]
// (already causal-masked by the forward, so masked entries are zero and their
// gradient stays zero). Composes SoftmaxBackward with device GEMMs; the causal
// mask needs no special handling in the backward.
func AttentionCoreBackward(worker *device.Worker, q, k, v, p, dOut []float32, seq, hd int, scale float64) (AttentionGrads, error) {
	if seq <= 0 || hd <= 0 || len(q) != seq*hd || len(k) != seq*hd || len(v) != seq*hd || len(p) != seq*seq || len(dOut) != seq*hd {
		return AttentionGrads{}, fmt.Errorf("AttentionCoreBackward: shape mismatch (seq=%d hd=%d)", seq, hd)
	}
	// dp = dOut·Vᵀ  [seq,hd]·[hd,seq] -> [seq,seq].
	dp, err := deviceGEMM(worker, false, true, seq, hd, seq, dOut, v)
	if err != nil {
		return AttentionGrads{}, err
	}
	// dscores = softmax_backward(p, dp), row-wise over keys.
	dscores, err := SoftmaxBackward(worker, p, dp, seq, seq)
	if err != nil {
		return AttentionGrads{}, err
	}
	// dV = pᵀ·dOut  [seq,seq]ᵀ·[seq,hd] -> [seq,hd].
	dV, err := deviceGEMM(worker, true, false, seq, seq, hd, p, dOut)
	if err != nil {
		return AttentionGrads{}, err
	}
	// dQ = scale·(dscores·K)  [seq,seq]·[seq,hd] -> [seq,hd].
	dQ, err := deviceGEMM(worker, false, false, seq, seq, hd, dscores, k)
	if err != nil {
		return AttentionGrads{}, err
	}
	// dK = scale·(dscoresᵀ·Q)  [seq,seq]ᵀ·[seq,hd] -> [seq,hd].
	dK, err := deviceGEMM(worker, true, false, seq, seq, hd, dscores, q)
	if err != nil {
		return AttentionGrads{}, err
	}
	s := float32(scale)
	for i := range dQ {
		dQ[i] *= s
		dK[i] *= s
	}
	return AttentionGrads{DQ: dQ, DK: dK, DV: dV, DScores: dscores}, nil
}
