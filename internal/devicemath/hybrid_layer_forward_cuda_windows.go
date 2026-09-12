//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/hostmath"
)

// HybridLayerDeviceCache holds host-side hybrid decoder forward intermediates
// until the matching backward consumes them. This is not device activation
// residency. The fields mirror the
// fields hostmath.hybridLayerCache carries and matching what
// standalone HybridDecoderLayerBackwardDevice prepares: the two pre-norm outputs
// and the mid residual (Xn/Hn/H), the active mix's forward cache
// (Attn xor GDN per IsLinear), and the SwiGLU MLP intermediates (GateP/UpP =
// Gate·hn / Up·hn, AP = SiLU(GateP), HMLP = AP*UpP).
type HybridLayerDeviceCache struct {
	dims      hostmath.HybridLayerDims
	ready     bool
	Xn, H, Hn []float32
	IsLinear  bool
	Attn      attnMixDeviceCache
	GDN       gatedDeltaMixDeviceCache
	GateP     []float32
	UpP       []float32
	AP        []float32
	HMLP      []float32
}

// HybridDecoderLayerForwardDevice is the device forward of hostmath's qwen3.5
// hybrid decoder layer (hostmath.HybridDecoderLayerForward). It composes the
// same individually parity-verified device ops the backward already uses --
// RMSNormForward, LinearForwardT, SiLUGateForward for the two pre-norms/residuals
// and the SwiGLU MLP, and the mix forward (attentionMixForwardDevice for
// full_attention incl. partial rope, gatedDeltaMixForwardDevice for the GDN
// linear_attention mix) -- wired exactly as the host golden:
//
//	h   = x + Mix(RMSNorm(x, InputNorm))
//	out = h + MLP(RMSNorm(h, PostNorm))
//
// It returns the layer output and the forward cache (HybridLayerDeviceCache) the
// resident-loop backward consumes. state is the GDN input state (nil/ignored for
// attention layers). The recomputed activations match the host cache within the
// ops' fp32 parity, so `out` matches hostmath.HybridDecoderLayerForward to the
// ~1e-4 kernel-parity class.
func HybridDecoderLayerForwardDevice(worker *device.Worker, x []float32, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state []float32) ([]float32, HybridLayerDeviceCache, error) {
	return hybridLayerForwardW(worker, x, hybridHostMatW(w), w, d, state)
}

// hybridLayerForwardW is HybridDecoderLayerForwardDevice over resident-or-host
// matrix weights (mw: the MLP + active mix projection matrices). The norm/conv/
// scalar VECTOR weights stay host-owned via w. When mw carries resident device
// pointers, the whole layer forward re-uploads NO matrix weight -- the resident
// hybrid runStack's per-step no-weight-motion forward.
func hybridLayerForwardW(worker *device.Worker, x []float32, mw hybridMatW, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state []float32) ([]float32, HybridLayerDeviceCache, error) {
	cache, err := hybridLayerCacheW(worker, x, mw, w, d, state)
	if err != nil {
		return nil, cache, err
	}
	mlpOut, err := linearForwardTW(worker, cache.HMLP, mw.mlp.down, d.Tokens, d.Inter, d.Hidden)
	if err != nil {
		return nil, cache, err
	}
	out := make([]float32, len(x))
	for i := range out {
		out[i] = cache.H[i] + mlpOut[i]
	}
	return out, cache, nil
}

// hybridLayerCacheW is the common forward preparation for execution and VJP
// callers that lack a retained cache. It stops before the final MLP projection.
func hybridLayerCacheW(worker *device.Worker, x []float32, mw hybridMatW, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state []float32) (HybridLayerDeviceCache, error) {
	T, H := d.Tokens, d.Hidden
	var cache HybridLayerDeviceCache
	if T <= 0 || H <= 0 || len(x) != T*H {
		return cache, fmt.Errorf("hybridLayerForwardW: shape mismatch (T=%d H=%d x=%d)", T, H, len(x))
	}
	cache.IsLinear = w.IsLinear
	cache.dims = d
	cache.ready = true
	eps := d.Eps

	// h = x + Mix(RMSNorm(x, InputNorm))
	xn, err := RMSNormForward(worker, x, w.InputNorm, T, H, eps)
	if err != nil {
		return cache, err
	}
	cache.Xn = xn

	var mixOut []float32
	if w.IsLinear {
		gc, err := gatedDeltaMixForwardDeviceW(worker, xn, mw.gdn, w.GDN, d.GDN, state)
		if err != nil {
			return cache, err
		}
		cache.GDN = gc
		valDim := d.GDN.ValueHeads * d.GDN.HeadDim
		// mix output: gated·Woutᵀ (the one matmul the recompute stops short of).
		mixOut, err = linearForwardTW(worker, gc.gated, mw.gdn.wout, T, valDim, d.GDN.OutDim)
		if err != nil {
			return cache, err
		}
	} else {
		var ac attnMixDeviceCache
		mixOut, ac, err = attentionMixForwardDeviceW(worker, xn, mw.attn, w.Attn, d.Attn)
		if err != nil {
			return cache, err
		}
		cache.Attn = ac
	}

	h := make([]float32, T*H)
	for i := range h {
		h[i] = x[i] + mixOut[i]
	}
	cache.H = h

	// out = h + MLP(RMSNorm(h, PostNorm))
	hn, err := RMSNormForward(worker, h, w.PostNorm, T, H, eps)
	if err != nil {
		return cache, err
	}
	cache.Hn = hn

	gateP, err := linearForwardTW(worker, hn, mw.mlp.gate, T, H, d.Inter)
	if err != nil {
		return cache, err
	}
	upP, err := linearForwardTW(worker, hn, mw.mlp.up, T, H, d.Inter)
	if err != nil {
		return cache, err
	}
	aP, hMLP, err := SiLUGateForward(worker, gateP, upP)
	if err != nil {
		return cache, err
	}
	cache.GateP, cache.UpP, cache.AP, cache.HMLP = gateP, upP, aP, hMLP

	return cache, nil
}
