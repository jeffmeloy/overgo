//go:build windows

package devicemath

import (
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
)

// RoPEHalfForward applies split-half (HF-layout) rotary embedding to a
// [seq, nHeads*hd] tensor in place: each pair (i, i+hd/2) of every head row at
// position pos rotates by angle pos*invFreq[i]. invFreq has hd/2 entries. This
// is the forward whose VJP is RoPEHalfBackward; it feeds the device attention
// forward (q and k rotation).
func RoPEHalfForward(worker *device.Worker, x, invFreq []float32, seq, nHeads, hd int) ([]float32, error) {
	if seq <= 0 || nHeads <= 0 || hd <= 0 || hd%2 != 0 || len(x) != seq*nHeads*hd || len(invFreq) != hd/2 {
		return nil, fmt.Errorf("RoPEHalfForward: shape mismatch (seq=%d nHeads=%d hd=%d x=%d inv=%d)", seq, nHeads, hd, len(x), len(invFreq))
	}
	out := make([]float32, len(x))
	copy(out, x)
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("rope_half_f32")
		if err != nil {
			return err
		}
		xPtr, err := scope.upload(out)
		if err != nil {
			return err
		}
		invPtr, err := scope.upload(invFreq)
		if err != nil {
			return err
		}
		seqU, nhU, hdU := uint32(seq), uint32(nHeads), uint32(hd)
		total := uint32(seq * nHeads * (hd / 2))
		if err := scope.launch1D(fn, total,
			unsafe.Pointer(&xPtr), unsafe.Pointer(&invPtr),
			unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{out, xPtr})
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RoPEHalfBackward is the VJP of split-half (HF-layout) rotary embedding applied
// to a [seq, nHeads*hd] tensor: it rotates the cotangent gradient by the negated
// angle pos*invFreq[i]. Since RoPE is linear (an orthogonal rotation), this
// depends only on the gradient, not the forward input. invFreq has hd/2 entries.
func RoPEHalfBackward(worker *device.Worker, gradOutput, invFreq []float32, seq, nHeads, hd int) ([]float32, error) {
	if seq <= 0 || nHeads <= 0 || hd <= 0 || hd%2 != 0 || len(gradOutput) != seq*nHeads*hd || len(invFreq) != hd/2 {
		return nil, fmt.Errorf("RoPEHalfBackward: shape mismatch (seq=%d nHeads=%d hd=%d g=%d inv=%d)", seq, nHeads, hd, len(gradOutput), len(invFreq))
	}
	dx := make([]float32, len(gradOutput))
	copy(dx, gradOutput)
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("rope_half_backward_f32")
		if err != nil {
			return err
		}
		dxPtr, err := scope.upload(dx)
		if err != nil {
			return err
		}
		invPtr, err := scope.upload(invFreq)
		if err != nil {
			return err
		}

		seqU, nhU, hdU := uint32(seq), uint32(nHeads), uint32(hd)
		total := uint32(seq * nHeads * (hd / 2))
		if err := scope.launch1D(fn, total,
			unsafe.Pointer(&dxPtr), unsafe.Pointer(&invPtr),
			unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{dx, dxPtr})
	})
	if err != nil {
		return nil, err
	}
	return dx, nil
}
