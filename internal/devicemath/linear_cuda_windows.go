//go:build windows

// Package devicemath holds GPU (CUDA, cgo-free) counterparts to internal/hostmath
// -- the device backward operators the training loop composes. Each is gated
// against the host by a finite-difference grad-check, matching adaptive_new's
// operators_backward verification.
package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// linWeight is a matrix weight the device linear ops read: either a HOST slice
// (uploaded into the session per call) or a RESIDENT device pointer (uploaded
// ONCE elsewhere and read in place, never re-uploaded and never freed by the
// session). It is the single seam that lets one linear-op implementation serve
// both the host-weight-fed path (existing callers) and the fully-resident
// hybrid runStack (weights live in a persistent dW buffer across the K-step
// loop -- no per-step weight motion).
type linWeight struct {
	host []float32        // non-nil => upload this slice per call
	dev  driver.DevicePtr // used when host == nil: a resident buffer sub-pointer
}

// hostW wraps a host weight slice (upload-per-call semantics, the legacy path).
func hostW(w []float32) linWeight { return linWeight{host: w} }

// devW wraps a resident device weight pointer (uploaded once, read in place).
func devW(p driver.DevicePtr) linWeight { return linWeight{dev: p} }

// resolve returns the device pointer for w within the session: the pre-existing
// resident pointer (no upload, no session ownership) or a freshly uploaded copy
// of the host slice (session-owned, freed on close). want is the expected
// element count, checked for the host case.
func (w linWeight) resolve(s *cudaBLAS, want int) (driver.DevicePtr, error) {
	if w.host == nil {
		return w.dev, nil
	}
	if len(w.host) != want {
		return 0, fmt.Errorf("linWeight: host len %d != want %d", len(w.host), want)
	}
	return s.upload(w.host)
}

type linearWeightLayout uint8

const (
	linearInputOutput linearWeightLayout = iota
	linearOutputInput
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
	return linearBackward(worker, "LinearBackward", linearInputOutput, x, w, dY, rows, in, out)
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
	return linearBackward(worker, "LinearBackwardT", linearOutputInput, x, w, dY, rows, in, outDim)
}

// LinearBackwardTResident writes the HF-layout VJP between resident buffers.
func LinearBackwardTResident(
	worker *device.Worker,
	x, weight, dY, dX, dWeight driver.DevicePtr,
	rows, in, out int,
) error {
	if worker == nil || x == 0 || weight == 0 || dY == 0 || dX == 0 || dWeight == 0 || rows <= 0 || in <= 0 || out <= 0 {
		return fmt.Errorf("LinearBackwardTResident: invalid buffer or geometry")
	}
	return withCUDABLAS(worker, func(session *cudaBLAS) error {
		if err := session.gemm(false, false, rows, out, in, dY, weight, dX); err != nil {
			return err
		}
		if err := session.gemm(true, false, out, rows, in, dY, x, dWeight); err != nil {
			return err
		}
		return session.finish()
	})
}

func linearBackward(
	worker *device.Worker,
	operator string,
	layout linearWeightLayout,
	x, w, dY []float32,
	rows, in, out int,
) (dX, dW []float32, err error) {
	if rows <= 0 || in <= 0 || out <= 0 || len(x) != rows*in || len(w) != in*out || len(dY) != rows*out {
		return nil, nil, fmt.Errorf("%s: shape mismatch (rows=%d in=%d out=%d x=%d w=%d dY=%d)", operator, rows, in, out, len(x), len(w), len(dY))
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
		if layout == linearOutputInput {
			if err := session.gemm(false, false, rows, out, in, dyPtr, wPtr, dxPtr); err != nil {
				return err
			}
			if err := session.gemm(true, false, out, rows, in, dyPtr, xPtr, dwPtr); err != nil {
				return err
			}
		} else {
			if err := session.gemm(false, true, rows, out, in, dyPtr, wPtr, dxPtr); err != nil {
				return err
			}
			if err := session.gemm(true, false, in, rows, out, xPtr, dyPtr, dwPtr); err != nil {
				return err
			}
		}
		return session.finish(cudaDownload{dX, dxPtr}, cudaDownload{dW, dwPtr})
	})
	if err != nil {
		return nil, nil, err
	}
	return dX, dW, nil
}

// LinearForwardT: HF-layout Y = X·Wᵀ via one cuBLAS GEMM.
func LinearForwardT(worker *device.Worker, x, w []float32, rows, in, out int) ([]float32, error) {
	if rows <= 0 || in <= 0 || out <= 0 || len(x) != rows*in || len(w) != out*in {
		return nil, fmt.Errorf("LinearForwardT: shape mismatch (rows=%d in=%d out=%d x=%d w=%d)", rows, in, out, len(x), len(w))
	}
	return linearForwardTW(worker, x, hostW(w), rows, in, out)
}

// linearForwardTW is LinearForwardT over a linWeight: the weight is uploaded
// per call (host) or read from its resident device pointer (dev). x uploads and
// the result downloads every call regardless -- only the WEIGHT residency
// differs, which is the whole point (a resident-weight forward moves no weight).
func linearForwardTW(worker *device.Worker, x []float32, w linWeight, rows, in, out int) ([]float32, error) {
	if rows <= 0 || in <= 0 || out <= 0 || len(x) != rows*in {
		return nil, fmt.Errorf("linearForwardTW: shape mismatch (rows=%d in=%d out=%d x=%d)", rows, in, out, len(x))
	}
	c := make([]float32, rows*out)
	err := withCUDABLAS(worker, func(s *cudaBLAS) error {
		xPtr, err := s.upload(x)
		if err != nil {
			return err
		}
		wPtr, err := w.resolve(s, out*in)
		if err != nil {
			return err
		}
		cPtr, err := s.alloc(len(c))
		if err != nil {
			return err
		}
		if err := s.gemm(false, true, rows, in, out, xPtr, wPtr, cPtr); err != nil {
			return err
		}
		return s.finish(cudaDownload{c, cPtr})
	})
	if err != nil {
		return nil, err
	}
	return c, nil
}

// linearBackwardTW is LinearBackwardT over a linWeight (HF layout, W stored
// [out,in]): dX = dY·W [rows,in], dW = dYᵀ·X [out,in]. The two GEMMs share one
// session; x and dY upload, the weight uploads (host) or is read in place
// (resident dev), and dX/dW come back. Same math as linearBackward's
// linearOutputInput branch; the resident-weight path just skips the weight
// upload.
func linearBackwardTW(worker *device.Worker, x []float32, w linWeight, dY []float32, rows, in, outDim int) (dX, dW []float32, err error) {
	if rows <= 0 || in <= 0 || outDim <= 0 || len(x) != rows*in || len(dY) != rows*outDim {
		return nil, nil, fmt.Errorf("linearBackwardTW: shape mismatch (rows=%d in=%d out=%d x=%d dY=%d)", rows, in, outDim, len(x), len(dY))
	}
	dX = make([]float32, rows*in)
	dW = make([]float32, outDim*in)
	err = withCUDABLAS(worker, func(s *cudaBLAS) error {
		xPtr, err := s.upload(x)
		if err != nil {
			return err
		}
		wPtr, err := w.resolve(s, outDim*in)
		if err != nil {
			return err
		}
		dyPtr, err := s.upload(dY)
		if err != nil {
			return err
		}
		dxPtr, err := s.alloc(len(dX))
		if err != nil {
			return err
		}
		dwPtr, err := s.alloc(len(dW))
		if err != nil {
			return err
		}
		if err := s.gemm(false, false, rows, outDim, in, dyPtr, wPtr, dxPtr); err != nil {
			return err
		}
		if err := s.gemm(true, false, outDim, rows, in, dyPtr, xPtr, dwPtr); err != nil {
			return err
		}
		return s.finish(cudaDownload{dX, dxPtr}, cudaDownload{dW, dwPtr})
	})
	if err != nil {
		return nil, nil, err
	}
	return dX, dW, nil
}
