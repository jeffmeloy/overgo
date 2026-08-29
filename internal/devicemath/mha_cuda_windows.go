//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
)

// extractHead copies head `head`'s [seq,hd] slice out of an interleaved
// [seq, nHeads*hd] tensor into a contiguous buffer.
func extractHead(src []float32, seq, nHeads, hd, head int) []float32 {
	out := make([]float32, seq*hd)
	for i := range seq {
		copy(out[i*hd:(i+1)*hd], src[i*nHeads*hd+head*hd:i*nHeads*hd+(head+1)*hd])
	}
	return out
}

// insertHead writes (or accumulates) a contiguous [seq,hd] head buffer back into
// an interleaved [seq, nHeads*hd] tensor at head `head`.
func insertHead(dst, headData []float32, seq, nHeads, hd, head int, accumulate bool) {
	for i := range seq {
		for j := range hd {
			if accumulate {
				dst[i*nHeads*hd+head*hd+j] += headData[i*hd+j]
			} else {
				dst[i*nHeads*hd+head*hd+j] = headData[i*hd+j]
			}
		}
	}
}

// MultiHeadAttentionBackward computes dQ/dK/dV for multi-head causal attention
// with grouped-query attention (nkv <= nh KV heads). Q is [seq, nh*hd]; K and V
// are [seq, nkv*hd]; p is the per-head causal softmax [nh, seq, seq]; dOut is
// [seq, nh*hd]. Each query head runs the verified single-head AttentionCoreBackward;
// query heads sharing a KV head accumulate into dK/dV. Correctness-first: heads
// run sequentially, each taking its own device context (a batched version is a
// later optimization).
func MultiHeadAttentionBackward(worker *device.Worker, q, k, v, p, dOut []float32, seq, nh, nkv, hd int, scale float64) (dQ, dK, dV []float32, err error) {
	if seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 || nh%nkv != 0 ||
		len(q) != seq*nh*hd || len(k) != seq*nkv*hd || len(v) != seq*nkv*hd ||
		len(p) != nh*seq*seq || len(dOut) != seq*nh*hd {
		return nil, nil, nil, fmt.Errorf("MultiHeadAttentionBackward: shape mismatch (seq=%d nh=%d nkv=%d hd=%d)", seq, nh, nkv, hd)
	}
	dQ = make([]float32, seq*nh*hd)
	dK = make([]float32, seq*nkv*hd)
	dV = make([]float32, seq*nkv*hd)
	headsPerKV := nh / nkv
	for h := range nh {
		kvh := h / headsPerKV
		qh := extractHead(q, seq, nh, hd, h)
		kh := extractHead(k, seq, nkv, hd, kvh)
		vh := extractHead(v, seq, nkv, hd, kvh)
		dOuth := extractHead(dOut, seq, nh, hd, h)
		ph := p[h*seq*seq : (h+1)*seq*seq]
		grads, err := AttentionCoreBackward(worker, qh, kh, vh, ph, dOuth, seq, hd, scale)
		if err != nil {
			return nil, nil, nil, err
		}
		insertHead(dQ, grads.DQ, seq, nh, hd, h, false)
		insertHead(dK, grads.DK, seq, nkv, hd, kvh, true) // grouped: accumulate
		insertHead(dV, grads.DV, seq, nkv, hd, kvh, true)
	}
	return dQ, dK, dV, nil
}
