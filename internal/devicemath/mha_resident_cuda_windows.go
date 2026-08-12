//go:build windows

package devicemath

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

// toHeadMajor reshapes [seq, nHeads*hd] -> [nHeads, seq, hd] (each head
// contiguous) so per-head device GEMMs address contiguous sub-buffers.
func toHeadMajor(x []float32, seq, nHeads, hd int) []float32 {
	out := make([]float32, len(x))
	for i := 0; i < seq; i++ {
		for h := 0; h < nHeads; h++ {
			copy(out[(h*seq+i)*hd:(h*seq+i+1)*hd], x[(i*nHeads+h)*hd:(i*nHeads+h+1)*hd])
		}
	}
	return out
}

// fromHeadMajor is the inverse of toHeadMajor.
func fromHeadMajor(x []float32, seq, nHeads, hd int) []float32 {
	out := make([]float32, len(x))
	for h := 0; h < nHeads; h++ {
		for i := 0; i < seq; i++ {
			copy(out[(i*nHeads+h)*hd:(i*nHeads+h+1)*hd], x[(h*seq+i)*hd:(h*seq+i+1)*hd])
		}
	}
	return out
}

// MultiHeadAttentionBackwardResident is the resident counterpart to
// MultiHeadAttentionBackward: the whole causal GQA attention-core backward runs
// in ONE worker.Do -- inputs reshaped head-major and uploaded once, every head's
// GEMMs and softmax_backward run on resident sub-buffers, GQA dk/dv accumulate
// on device via the add kernel, and only dQ/dK/dV come back. The softmax
// probabilities are recomputed on device per head (causal_softmax of qh·khᵀ)
// rather than received as a host [nh, seq, seq] slice -- the score scale is
// folded into q upstream, so scores use scale 1 (dQ/dK are scaled on return).
// Same math as MultiHeadAttentionBackward. Addresses SQA finding 2 for the
// attention block and removes the O(heads*seq^2) host softmax allocation.
func MultiHeadAttentionBackwardResident(worker *device.Worker, q, k, v, dOut []float32, seq, nh, nkv, hd int, scale float64) (dQ, dK, dV []float32, err error) {
	if seq <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 || nh%nkv != 0 ||
		len(q) != seq*nh*hd || len(k) != seq*nkv*hd || len(v) != seq*nkv*hd ||
		len(dOut) != seq*nh*hd {
		return nil, nil, nil, fmt.Errorf("MultiHeadAttentionBackwardResident: shape mismatch (seq=%d nh=%d nkv=%d hd=%d)", seq, nh, nkv, hd)
	}
	group := nh / nkv
	qC := toHeadMajor(q, seq, nh, hd)
	kC := toHeadMajor(k, seq, nkv, hd)
	vC := toHeadMajor(v, seq, nkv, hd)
	dOutC := toHeadMajor(dOut, seq, nh, hd)
	dqC := make([]float32, seq*nh*hd)
	dkC := make([]float32, seq*nkv*hd)
	dvC := make([]float32, seq*nkv*hd)

	err = worker.Do(context.Background(), func(state *device.State) error {
		lib := state.Driver
		blas, err := cublas.Open()
		if err != nil {
			return err
		}
		defer blas.Close()
		handle, err := blas.Create()
		if err != nil {
			return err
		}
		defer blas.Destroy(handle)
		if err := blas.SetStream(handle, state.Stream); err != nil {
			return err
		}
		module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return err
		}
		defer lib.ModuleUnload(module)
		addFn, err := lib.ModuleFunction(module, "add_f32")
		if err != nil {
			return err
		}
		softmaxFn, err := lib.ModuleFunction(module, "softmax_backward_f32")
		if err != nil {
			return err
		}
		causalSoftmaxFn, err := lib.ModuleFunction(module, "causal_softmax_f32")
		if err != nil {
			return err
		}

		var frees []driver.DevicePtr
		defer func() {
			for _, pp := range frees {
				lib.MemFree(pp)
			}
		}()
		alloc := func(n int) (driver.DevicePtr, error) {
			pp, err := lib.MemAlloc(uint64(n) * 4)
			if err != nil {
				return 0, err
			}
			frees = append(frees, pp)
			return pp, nil
		}
		upload := func(data []float32) (driver.DevicePtr, error) {
			pp, err := alloc(len(data))
			if err != nil {
				return 0, err
			}
			return pp, lib.MemcpyHtoD(pp, driver.Bytes(data))
		}

		qP, err := upload(qC)
		if err != nil {
			return err
		}
		kP, err := upload(kC)
		if err != nil {
			return err
		}
		vP, err := upload(vC)
		if err != nil {
			return err
		}
		dOutP, err := upload(dOutC)
		if err != nil {
			return err
		}
		dqP, err := alloc(seq * nh * hd)
		if err != nil {
			return err
		}
		dkP, err := alloc(seq * nkv * hd)
		if err != nil {
			return err
		}
		dvP, err := alloc(seq * nkv * hd)
		if err != nil {
			return err
		}
		// dk/dv accumulate across the group -> zero first.
		if err := lib.MemsetD32Async(dkP, 0, uint64(seq*nkv*hd), state.Stream); err != nil {
			return err
		}
		if err := lib.MemsetD32Async(dvP, 0, uint64(seq*nh*hd/group), state.Stream); err != nil {
			return err
		}
		dpP, err := alloc(seq * seq)
		if err != nil {
			return err
		}
		dsP, err := alloc(seq * seq)
		if err != nil {
			return err
		}
		// Per-head score + softmax scratch (reused each head): scores = qh·khᵀ,
		// then causal_softmax -> phP.
		scoresP, err := alloc(seq * seq)
		if err != nil {
			return err
		}
		phP, err := alloc(seq * seq)
		if err != nil {
			return err
		}
		tmpP, err := alloc(seq * hd)
		if err != nil {
			return err
		}

		off := func(base driver.DevicePtr, elems int) driver.DevicePtr {
			return base + driver.DevicePtr(elems*4)
		}
		launch := func(fn driver.Function, n int, args []unsafe.Pointer) error {
			const threads = uint32(256)
			blocks := (uint32(n) + threads - 1) / threads
			return lib.LaunchKernel(fn, driver.Dim3{X: blocks, Y: 1, Z: 1}, driver.Dim3{X: threads, Y: 1, Z: 1}, 0, state.Stream, args)
		}
		add := func(a, b, out driver.DevicePtr, n int) error {
			c := uint32(n)
			e := launch(addFn, n, []unsafe.Pointer{unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&out), unsafe.Pointer(&c)})
			runtime.KeepAlive(a)
			runtime.KeepAlive(b)
			runtime.KeepAlive(out)
			return e
		}

		hs := seq * hd
		for h := 0; h < nh; h++ {
			kv := h / group
			qh, dOuth, dqh := off(qP, h*hs), off(dOutP, h*hs), off(dqP, h*hs)
			kh, vh, dkh, dvh := off(kP, kv*hs), off(vP, kv*hs), off(dkP, kv*hs), off(dvP, kv*hs)

			// p_h = causal_softmax(qh·khᵀ) on device (scores use scale 1, folded
			// into q upstream) -- replaces the uploaded host softmax slice.
			if err := blas.RowMajorGEMMExF32(handle, false, true, int32(seq), int32(hd), int32(seq), qh, kh, scoresP); err != nil {
				return err
			}
			rowsCausal := uint32(seq)
			if err := launch(causalSoftmaxFn, seq, []unsafe.Pointer{unsafe.Pointer(&scoresP), unsafe.Pointer(&phP), unsafe.Pointer(&rowsCausal)}); err != nil {
				return err
			}
			runtime.KeepAlive(scoresP)
			ph := phP

			// dp = dOuth·vhᵀ  [seq,seq]
			if err := blas.RowMajorGEMMExF32(handle, false, true, int32(seq), int32(hd), int32(seq), dOuth, vh, dpP); err != nil {
				return err
			}
			// dscores = softmax_backward(ph, dp)  [seq rows of length seq]
			rowsU, dU := uint32(seq), uint32(seq)
			if err := launch(softmaxFn, seq, []unsafe.Pointer{unsafe.Pointer(&ph), unsafe.Pointer(&dpP), unsafe.Pointer(&dsP), unsafe.Pointer(&rowsU), unsafe.Pointer(&dU)}); err != nil {
				return err
			}
			runtime.KeepAlive(ph)
			// dV_h = phᵀ·dOuth  -> accumulate into dvh
			if err := blas.RowMajorGEMMExF32(handle, true, false, int32(seq), int32(seq), int32(hd), ph, dOuth, tmpP); err != nil {
				return err
			}
			if err := add(dvh, tmpP, dvh, hs); err != nil {
				return err
			}
			// dQ_h = dscores·kh  -> unique head, write
			if err := blas.RowMajorGEMMExF32(handle, false, false, int32(seq), int32(seq), int32(hd), dsP, kh, dqh); err != nil {
				return err
			}
			// dK_h = dscoresᵀ·qh  -> accumulate into dkh
			if err := blas.RowMajorGEMMExF32(handle, true, false, int32(seq), int32(seq), int32(hd), dsP, qh, tmpP); err != nil {
				return err
			}
			if err := add(dkh, tmpP, dkh, hs); err != nil {
				return err
			}
		}

		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		if err := lib.MemcpyDtoH(driver.Bytes(dqC), dqP); err != nil {
			return err
		}
		if err := lib.MemcpyDtoH(driver.Bytes(dkC), dkP); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(dvC), dvP)
	})
	if err != nil {
		return nil, nil, nil, err
	}
	dQ = fromHeadMajor(dqC, seq, nh, hd)
	dK = fromHeadMajor(dkC, seq, nkv, hd)
	dV = fromHeadMajor(dvC, seq, nkv, hd)
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
