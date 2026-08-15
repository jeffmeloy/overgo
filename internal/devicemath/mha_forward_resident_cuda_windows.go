//go:build windows

package devicemath

import (
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// MultiHeadAttentionForwardResident: one-session causal GQA forward.
// Expects q-scaled scores; returns attention output only.
func MultiHeadAttentionForwardResident(worker *device.Worker, q, k, v []float32, seq, nh, nkv, hd int) ([]float32, error) {
	if seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 || nh%nkv != 0 ||
		len(q) != seq*nh*hd || len(k) != seq*nkv*hd || len(v) != seq*nkv*hd {
		return nil, fmt.Errorf("MultiHeadAttentionForwardResident: shape mismatch (seq=%d nh=%d nkv=%d hd=%d)", seq, nh, nkv, hd)
	}
	group := nh / nkv
	qC := headMajorLayout(q, seq, nh, hd, headMajorPack)
	kC := headMajorLayout(k, seq, nkv, hd, headMajorPack)
	vC := headMajorLayout(v, seq, nkv, hd, headMajorPack)
	attnCoreC := make([]float32, seq*nh*hd)

	err := withCUDABLAS(worker, func(session *cudaBLAS) error {
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
		attnP, err := session.alloc(seq * nh * hd)
		if err != nil {
			return err
		}
		// Per-head score + softmax scratch, reused each head.
		scoresP, err := session.alloc(seq * seq)
		if err != nil {
			return err
		}
		phP, err := session.alloc(seq * seq)
		if err != nil {
			return err
		}

		off := func(base driver.DevicePtr, elems int) driver.DevicePtr {
			return base + driver.DevicePtr(uint64(elems)*f32Bytes)
		}
		hs := seq * hd
		for h := 0; h < nh; h++ {
			kv := h / group
			qh := off(qP, h*hs)
			kh, vh := off(kP, kv*hs), off(vP, kv*hs)
			attnh := off(attnP, h*hs)

			// scores = qh·khᵀ [seq,seq]
			if err := session.gemm(false, true, seq, hd, seq, qh, kh, scoresP); err != nil {
				return err
			}
			// p = causal_softmax(scores) [seq,seq]
			rowsCausal := uint32(seq)
			if err := session.launch1D(causalSoftmaxFn, rowsCausal,
				unsafe.Pointer(&scoresP), unsafe.Pointer(&phP), unsafe.Pointer(&rowsCausal)); err != nil {
				return err
			}
			// attnCore_h = p·vh [seq,hd]
			if err := session.gemm(false, false, seq, seq, hd, phP, vh, attnh); err != nil {
				return err
			}
		}
		return session.finish(cudaDownload{attnCoreC, attnP})
	})
	if err != nil {
		return nil, err
	}
	return headMajorLayout(attnCoreC, seq, nh, hd, headMajorUnpack), nil
}
