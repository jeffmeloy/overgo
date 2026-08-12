//go:build windows

package devicemath

import (
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// MultiHeadAttentionForwardResident is the forward counterpart to
// MultiHeadAttentionBackwardResident: the causal GQA attention core runs in ONE
// cudaBLAS session -- q/k/v reshaped head-major and uploaded once, and per head
// the scores (qh·khᵀ), the causal softmax (causal_softmax_f32), and
// attnCore = p·v run on resident sub-buffers. Only attnCore [seq, nh*hd] comes
// back. The score scale is assumed folded into q upstream (scores use scale 1),
// matching hostmath.CausalAttention and the resident backward. This is the
// attention piece of the device forward that will feed the resident layer cache.
func MultiHeadAttentionForwardResident(worker *device.Worker, q, k, v []float32, seq, nh, nkv, hd int) ([]float32, error) {
	if seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 || nh%nkv != 0 ||
		len(q) != seq*nh*hd || len(k) != seq*nkv*hd || len(v) != seq*nkv*hd {
		return nil, fmt.Errorf("MultiHeadAttentionForwardResident: shape mismatch (seq=%d nh=%d nkv=%d hd=%d)", seq, nh, nkv, hd)
	}
	group := nh / nkv
	qC := toHeadMajor(q, seq, nh, hd)
	kC := toHeadMajor(k, seq, nkv, hd)
	vC := toHeadMajor(v, seq, nkv, hd)
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
	return fromHeadMajor(attnCoreC, seq, nh, hd), nil
}
