//go:build windows

// Package devicemath holds GPU (CUDA, cgo-free) counterparts to internal/hostmath
// -- the device backward operators the training loop composes. Each is gated
// against the host by a finite-difference grad-check, matching adaptive_new's
// operators_backward verification.
package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
)

// LinearBackward computes the gradients of the row-major linear map
// Y = X·W (X[rows,in], W[in,out], Y[rows,out]) given the output cotangent
// dY[rows,out], on the GPU in fp32:
//
//	dX = dY·Wᵀ   [rows,in]
//	dW = Xᵀ·dY   [in,out]
//
// Both are cuBLAS GEMMs (no custom kernel). This is the fundamental device
// backward operator; linear/attention/MLP backward compose it.
func LinearBackward(worker *device.Worker, x, w, dY []float32, rows, in, out int) (dX, dW []float32, err error) {
	if rows <= 0 || in <= 0 || out <= 0 || len(x) != rows*in || len(w) != in*out || len(dY) != rows*out {
		return nil, nil, fmt.Errorf("LinearBackward: shape mismatch (rows=%d in=%d out=%d x=%d w=%d dY=%d)", rows, in, out, len(x), len(w), len(dY))
	}
	dX = make([]float32, rows*in)
	dW = make([]float32, in*out)
	err = withCUDABLAS(worker, func(session *cudaBLAS) error {
		xPtr, err := session.upload(x)
		if err != nil {
			return err
		}
		wPtr, err := session.upload(w)
		if err != nil {
			return err
		}
		dyPtr, err := session.upload(dY)
		if err != nil {
			return err
		}
		dxPtr, err := session.alloc(len(dX))
		if err != nil {
			return err
		}
		dwPtr, err := session.alloc(len(dW))
		if err != nil {
			return err
		}
		if err := session.gemm(false, true, rows, out, in, dyPtr, wPtr, dxPtr); err != nil {
			return err
		}
		if err := session.gemm(true, false, in, rows, out, xPtr, dyPtr, dwPtr); err != nil {
			return err
		}
		return session.finish(cudaDownload{dX, dxPtr}, cudaDownload{dW, dwPtr})
	})
	if err != nil {
		return nil, nil, err
	}
	return dX, dW, nil
}

// LinearBackwardT computes the gradients of the HF-layout linear map
// Y = X·Wᵀ (X[rows,in], W[outDim,in], Y[rows,outDim]) -- densecausal's convention
// (hostmath.Linear), the transpose of LinearBackward's Y=X·W. Given dY:
//
//	dX = dY·W    [rows,in]
//	dW = dYᵀ·X   [outDim,in]
//
// The two GEMMs share one device context: x, w and dY are uploaded once (dY
// feeds both products), both cuBLAS calls run on resident device pointers, and
// only dX/dW come back -- one worker.Do per linear VJP instead of two. This is
// the linear adapter the densecausal-exact device layer backward uses for its
// q/k/v/o and gate/up/down projections, whose weights are stored [out,in].
func LinearBackwardT(worker *device.Worker, x, w, dY []float32, rows, in, outDim int) (dX, dW []float32, err error) {
	if rows <= 0 || in <= 0 || outDim <= 0 || len(x) != rows*in || len(w) != outDim*in || len(dY) != rows*outDim {
		return nil, nil, fmt.Errorf("LinearBackwardT: shape mismatch (rows=%d in=%d outDim=%d x=%d w=%d dY=%d)", rows, in, outDim, len(x), len(w), len(dY))
	}
	dX = make([]float32, rows*in)
	dW = make([]float32, outDim*in)
	err = withCUDABLAS(worker, func(session *cudaBLAS) error {
		xPtr, err := session.upload(x)
		if err != nil {
			return err
		}
		wPtr, err := session.upload(w)
		if err != nil {
			return err
		}
		dyPtr, err := session.upload(dY)
		if err != nil {
			return err
		}
		dxPtr, err := session.alloc(len(dX))
		if err != nil {
			return err
		}
		dwPtr, err := session.alloc(len(dW))
		if err != nil {
			return err
		}
		if err := session.gemm(false, false, rows, outDim, in, dyPtr, wPtr, dxPtr); err != nil {
			return err
		}
		if err := session.gemm(true, false, outDim, rows, in, dyPtr, xPtr, dwPtr); err != nil {
			return err
		}
		return session.finish(cudaDownload{dX, dxPtr}, cudaDownload{dW, dwPtr})
	})
	if err != nil {
		return nil, nil, err
	}
	return dX, dW, nil
}
