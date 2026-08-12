//go:build windows

package devicemath

import (
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
)

// ShortConvBackwardDevice is the cgo-free CUDA port of hostmath.ShortConvBackward:
// the VJP of SiLU(depthwise causal conv1d, left-pad k-1) -- the qwen3.5 GDN-mix
// short convolution. x, dY and dX are [channels, tokens] channel-major; w and dW
// are [channels, k]; bias/dBias are per channel (bias nil => dBias nil). Launches
// short_conv_backward_f32 with one thread per channel (channels independent); a
// f64 scratch buffer stages the per-token SiLU'(pre)*dY product reused by the dW
// and dX passes, all accumulated in double to mirror the host f64 golden.
func ShortConvBackwardDevice(worker *device.Worker, x, dY []float32, channels, tokens int, w, bias []float32, k int) (dX, dW, dBias []float32, err error) {
	if channels <= 0 || tokens <= 0 || k <= 0 {
		return nil, nil, nil, fmt.Errorf("ShortConvBackwardDevice: nonpositive dimension (channels=%d tokens=%d k=%d)", channels, tokens, k)
	}
	if len(x) != channels*tokens || len(dY) != channels*tokens || len(w) != channels*k {
		return nil, nil, nil, fmt.Errorf("ShortConvBackwardDevice: shape mismatch (channels=%d tokens=%d k=%d x=%d dY=%d w=%d)", channels, tokens, k, len(x), len(dY), len(w))
	}
	hasBias := bias != nil
	if hasBias && len(bias) != channels {
		return nil, nil, nil, fmt.Errorf("ShortConvBackwardDevice: bias length %d != channels %d", len(bias), channels)
	}

	dX = make([]float32, channels*tokens)
	dW = make([]float32, channels*k)
	dBiasBuf := make([]float32, channels) // always alloc; downloaded only when hasBias
	biasUpload := bias
	if !hasBias {
		biasUpload = make([]float32, channels) // dummy, unread by the kernel
	}

	err = withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("short_conv_backward_f32")
		if err != nil {
			return err
		}
		xPtr, err := scope.upload(x)
		if err != nil {
			return err
		}
		dyPtr, err := scope.upload(dY)
		if err != nil {
			return err
		}
		wPtr, err := scope.upload(w)
		if err != nil {
			return err
		}
		bPtr, err := scope.upload(biasUpload)
		if err != nil {
			return err
		}
		dxPtr, err := scope.alloc(channels * tokens)
		if err != nil {
			return err
		}
		dwPtr, err := scope.alloc(channels * k)
		if err != nil {
			return err
		}
		dbPtr, err := scope.alloc(channels)
		if err != nil {
			return err
		}
		scratchPtr, err := scope.allocRaw(uint64(channels) * uint64(tokens) * doubleBytes)
		if err != nil {
			return err
		}
		chU, tU, kU := uint32(channels), uint32(tokens), uint32(k)
		hasB := uint32(0)
		if hasBias {
			hasB = 1
		}
		if err := scope.launch1D(fn, chU,
			unsafe.Pointer(&xPtr), unsafe.Pointer(&dyPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&bPtr),
			unsafe.Pointer(&dxPtr), unsafe.Pointer(&dwPtr), unsafe.Pointer(&dbPtr), unsafe.Pointer(&scratchPtr),
			unsafe.Pointer(&chU), unsafe.Pointer(&tU), unsafe.Pointer(&kU), unsafe.Pointer(&hasB),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{dX, dxPtr}, cudaDownload{dW, dwPtr}, cudaDownload{dBiasBuf, dbPtr})
	})
	if err != nil {
		return nil, nil, nil, err
	}
	if hasBias {
		dBias = dBiasBuf
	}
	return dX, dW, dBias, nil
}
