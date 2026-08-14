//go:build windows

package devicemath

import (
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
)

type ropeHalfDirection uint8

const (
	ropeHalfForward ropeHalfDirection = iota
	ropeHalfBackward
)

// RoPEHalfForward applies split-half (HF-layout) rotary embedding to a
// [seq, nHeads*hd] tensor in place: each pair (i, i+hd/2) of every head row at
// position pos rotates by angle pos*invFreq[i]. invFreq has hd/2 entries. This
// is the forward whose VJP is RoPEHalfBackward; it feeds the device attention
// forward (q and k rotation).
func RoPEHalfForward(worker *device.Worker, x, invFreq []float32, seq, nHeads, hd int) ([]float32, error) {
	return applyRoPEHalf(worker, ropeHalfForward, x, invFreq, seq, nHeads, hd)
}

// RoPEHalfBackward is the VJP of split-half (HF-layout) rotary embedding applied
// to a [seq, nHeads*hd] tensor: it rotates the cotangent gradient by the negated
// angle pos*invFreq[i]. Since RoPE is linear (an orthogonal rotation), this
// depends only on the gradient, not the forward input. invFreq has hd/2 entries.
func RoPEHalfBackward(worker *device.Worker, gradOutput, invFreq []float32, seq, nHeads, hd int) ([]float32, error) {
	return applyRoPEHalf(worker, ropeHalfBackward, gradOutput, invFreq, seq, nHeads, hd)
}

func applyRoPEHalf(
	worker *device.Worker,
	direction ropeHalfDirection,
	input, invFreq []float32,
	seq, nHeads, hd int,
) ([]float32, error) {
	operator, kernel := "RoPEHalfForward", "rope_half_f32"
	if direction == ropeHalfBackward {
		operator, kernel = "RoPEHalfBackward", "rope_half_backward_f32"
	}
	if seq <= 0 || nHeads <= 0 || hd <= 0 || hd%2 != 0 || len(input) != seq*nHeads*hd || len(invFreq) != hd/2 {
		return nil, fmt.Errorf("%s: shape mismatch (seq=%d nHeads=%d hd=%d input=%d inv=%d)", operator, seq, nHeads, hd, len(input), len(invFreq))
	}
	output := append([]float32(nil), input...)
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function(kernel)
		if err != nil {
			return err
		}
		outputPtr, err := scope.upload(output)
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
			unsafe.Pointer(&outputPtr), unsafe.Pointer(&invPtr),
			unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{output, outputPtr})
	})
	if err != nil {
		return nil, err
	}
	return output, nil
}
