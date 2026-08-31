//go:build windows

package devicemath

import (
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

type headMajorDirection string

const (
	headMajorPack   headMajorDirection = "head_major_f32"
	headMajorUnpack headMajorDirection = "head_major_inverse_f32"
)

func headMajorLayout(x []float32, seq, nHeads, hd int, direction headMajorDirection) []float32 {
	sourceSeq, sourceHead := nHeads*hd, hd
	targetSeq, targetHead := hd, seq*hd
	if direction == headMajorUnpack {
		sourceSeq, targetSeq = targetSeq, sourceSeq
		sourceHead, targetHead = targetHead, sourceHead
	}
	out := make([]float32, len(x))
	for head := range nHeads {
		for row := range seq {
			source := row*sourceSeq + head*sourceHead
			target := row*targetSeq + head*targetHead
			copy(out[target:target+hd], x[source:source+hd])
		}
	}
	return out
}

// MultiHeadAttentionBackwardResident: one-session causal GQA backward.
// Recomputes softmax on device; returns dQ/dK/dV only.
func MultiHeadAttentionBackwardResident(worker *device.Worker, q, k, v, dOut []float32, seq, nh, nkv, hd int, scale float64) (dQ, dK, dV []float32, err error) {
	if seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 || nh%nkv != 0 ||
		len(q) != seq*nh*hd || len(k) != seq*nkv*hd || len(v) != seq*nkv*hd ||
		len(dOut) != seq*nh*hd {
		return nil, nil, nil, fmt.Errorf("MultiHeadAttentionBackwardResident: shape mismatch (seq=%d nh=%d nkv=%d hd=%d)", seq, nh, nkv, hd)
	}
	group := nh / nkv
	qC := headMajorLayout(q, seq, nh, hd, headMajorPack)
	kC := headMajorLayout(k, seq, nkv, hd, headMajorPack)
	vC := headMajorLayout(v, seq, nkv, hd, headMajorPack)
	dOutC := headMajorLayout(dOut, seq, nh, hd, headMajorPack)
	dqC := make([]float32, seq*nh*hd)
	dkC := make([]float32, seq*nkv*hd)
	dvC := make([]float32, seq*nkv*hd)

	err = withCUDABLAS(worker, func(session *cudaBLAS) error {
		addFn, err := session.function("add_f32")
		if err != nil {
			return err
		}
		softmaxFn, err := session.function("softmax_backward_f32")
		if err != nil {
			return err
		}
		causalSoftmaxFn, err := session.function("causal_softmax_f32")
		if err != nil {
			return err
		}

		qP, err := session.upload(qC)
		if err != nil {
			return err
		}
		kP, err := session.upload(kC)
		if err != nil {
			return err
		}
		vP, err := session.upload(vC)
		if err != nil {
			return err
		}
		dOutP, err := session.upload(dOutC)
		if err != nil {
			return err
		}
		dqP, err := session.alloc(seq * nh * hd)
		if err != nil {
			return err
		}
		dkP, err := session.alloc(seq * nkv * hd)
		if err != nil {
			return err
		}
		dvP, err := session.alloc(seq * nkv * hd)
		if err != nil {
			return err
		}
		// dk/dv accumulate across the group -> zero first.
		if err := session.state.Driver.MemsetD32Async(dkP, 0, uint64(seq*nkv*hd), session.state.Stream); err != nil {
			return err
		}
		if err := session.state.Driver.MemsetD32Async(dvP, 0, uint64(seq*nh*hd/group), session.state.Stream); err != nil {
			return err
		}
		dpP, err := session.alloc(seq * seq)
		if err != nil {
			return err
		}
		dsP, err := session.alloc(seq * seq)
		if err != nil {
			return err
		}
		// Per-head score + softmax scratch (reused each head): scores = qh·khᵀ,
		// then causal_softmax -> phP (device p, no host [nh,seq,seq] upload).
		scoresP, err := session.alloc(seq * seq)
		if err != nil {
			return err
		}
		phP, err := session.alloc(seq * seq)
		if err != nil {
			return err
		}
		tmpP, err := session.alloc(seq * hd)
		if err != nil {
			return err
		}

		off := func(base driver.DevicePtr, elems int) driver.DevicePtr {
			return base + driver.DevicePtr(uint64(elems)*f32Bytes)
		}
		add := func(a, b, out driver.DevicePtr, n int) error {
			return session.launchVector3(addFn, a, b, out, n)
		}

		hs := seq * hd
		for h := range nh {
			kv := h / group
			qh, dOuth, dqh := off(qP, h*hs), off(dOutP, h*hs), off(dqP, h*hs)
			kh, vh, dkh, dvh := off(kP, kv*hs), off(vP, kv*hs), off(dkP, kv*hs), off(dvP, kv*hs)

			// p_h = causal_softmax(qh·khᵀ) on device (scores use scale 1, folded
			// into q upstream) -- replaces the uploaded host softmax slice.
			if err := session.gemm(false, true, seq, hd, seq, qh, kh, scoresP); err != nil {
				return err
			}
			rowsCausal := uint32(seq)
			if err := session.launch1D(causalSoftmaxFn, rowsCausal,
				unsafe.Pointer(&scoresP), unsafe.Pointer(&phP), unsafe.Pointer(&rowsCausal)); err != nil {
				return err
			}
			ph := phP

			// dp = dOuth·vhᵀ  [seq,seq]
			if err := session.gemm(false, true, seq, hd, seq, dOuth, vh, dpP); err != nil {
				return err
			}
			// dscores = softmax_backward(ph, dp)  [seq rows of length seq]
			rowsU, dU := uint32(seq), uint32(seq)
			if err := session.launch1D(softmaxFn, rowsU,
				unsafe.Pointer(&ph), unsafe.Pointer(&dpP), unsafe.Pointer(&dsP),
				unsafe.Pointer(&rowsU), unsafe.Pointer(&dU)); err != nil {
				return err
			}
			// dV_h = phᵀ·dOuth  -> accumulate into dvh
			if err := session.gemm(true, false, seq, seq, hd, ph, dOuth, tmpP); err != nil {
				return err
			}
			if err := add(dvh, tmpP, dvh, hs); err != nil {
				return err
			}
			// dQ_h = dscores·kh  -> unique head, write
			if err := session.gemm(false, false, seq, seq, hd, dsP, kh, dqh); err != nil {
				return err
			}
			// dK_h = dscoresᵀ·qh  -> accumulate into dkh
			if err := session.gemm(true, false, seq, seq, hd, dsP, qh, tmpP); err != nil {
				return err
			}
			if err := add(dkh, tmpP, dkh, hs); err != nil {
				return err
			}
		}

		return session.finish(
			cudaDownload{dqC, dqP}, cudaDownload{dkC, dkP}, cudaDownload{dvC, dvP},
		)
	})
	if err != nil {
		return nil, nil, nil, err
	}
	dQ = headMajorLayout(dqC, seq, nh, hd, headMajorUnpack)
	dK = headMajorLayout(dkC, seq, nkv, hd, headMajorUnpack)
	dV = headMajorLayout(dvC, seq, nkv, hd, headMajorUnpack)
	// scores = (Q·Kᵀ)*scale, so scale multiplies both dQ and dK (matching
	// AttentionCoreBackward); dV is unscaled.
	if scale != 1 {
		s := float32(scale)
		for i := range dQ {
			dQ[i] *= s
		}
		for i := range dK {
			dK[i] *= s
		}
	}
	return dQ, dK, dV, nil
}
