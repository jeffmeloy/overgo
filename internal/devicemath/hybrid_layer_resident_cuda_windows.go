//go:build windows

package devicemath

import (
	"fmt"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/hostmath"
)

// attnMatW / gdnMatW bundle one layer's active-mix
// projection matrices as linWeights -- either host slices (upload-per-call) or
// resident device pointers (uploaded once, read in place). hybridMatW is the
// per-layer union the layer forward/backward thread; the mix branch used is
// selected by w.IsLinear, so only the matching bundle is populated.
type attnMatW struct{ wq, wk, wv, wo linWeight }
type gdnMatW struct{ wq, wk, wv, wbeta, walpha, wz, wout linWeight }

type hybridMatW struct {
	mlp  mlpMatW
	attn attnMatW
	gdn  gdnMatW
}

// Host matrix bundles: upload-per-call device operators.
func attnHostMatW(w hostmath.AttentionMixWeights) attnMatW {
	return attnMatW{wq: hostW(w.Wq), wk: hostW(w.Wk), wv: hostW(w.Wv), wo: hostW(w.Wo)}
}

func gdnHostMatW(w hostmath.GatedDeltaMixWeights) gdnMatW {
	return gdnMatW{
		wq: hostW(w.Wq), wk: hostW(w.Wk), wv: hostW(w.Wv),
		wbeta: hostW(w.Wbeta), walpha: hostW(w.Walpha), wz: hostW(w.Wz), wout: hostW(w.Wout),
	}
}

func hybridHostMatW(w hostmath.HybridLayerWeights) hybridMatW {
	m := hybridMatW{mlp: mlpMatW{gate: hostW(w.MLP.Gate), up: hostW(w.MLP.Up), down: hostW(w.MLP.Down)}}
	if w.IsLinear {
		m.gdn = gdnHostMatW(w.GDN)
	} else {
		m.attn = attnHostMatW(w.Attn)
	}
	return m
}

// HybridLayerResidentMatrices binds matrix views in a caller-owned weight or
// gradient slab. Both slabs use the caller's validated optimizer layout.
// Vector weights and gradients remain host-owned. Populate the active mix;
// views for the other mix are ignored.
type HybridLayerResidentMatrices struct {
	IsLinear bool
	// SwiGLU MLP matrices (resident).
	MLPGate, MLPUp, MLPDown driver.DevicePtr
	// full_attention matrices (resident); used when !IsLinear.
	AttnWq, AttnWk, AttnWv, AttnWo driver.DevicePtr
	// linear_attention (GDN) matrices (resident); used when IsLinear.
	GDNWq, GDNWk, GDNWv, GDNWbeta, GDNWalpha, GDNWz, GDNWout driver.DevicePtr
}

// matW converts the resident pointers into the linWeight bundle the shared layer
// forward/backward consume -- every matrix a devW (read in place, never uploaded
// or freed by the op session).
func (rw HybridLayerResidentMatrices) matW() hybridMatW {
	m := hybridMatW{mlp: mlpMatW{gate: devW(rw.MLPGate), up: devW(rw.MLPUp), down: devW(rw.MLPDown)}}
	if rw.IsLinear {
		m.gdn = gdnMatW{
			wq: devW(rw.GDNWq), wk: devW(rw.GDNWk), wv: devW(rw.GDNWv),
			wbeta: devW(rw.GDNWbeta), walpha: devW(rw.GDNWalpha), wz: devW(rw.GDNWz), wout: devW(rw.GDNWout),
		}
	} else {
		m.attn = attnMatW{wq: devW(rw.AttnWq), wk: devW(rw.AttnWk), wv: devW(rw.AttnWv), wo: devW(rw.AttnWo)}
	}
	return m
}

func (rw HybridLayerResidentMatrices) withGradients(gradients HybridLayerResidentMatrices) (hybridMatW, error) {
	weights := rw.matW()
	if rw.IsLinear != gradients.IsLinear {
		return hybridMatW{}, fmt.Errorf("hybrid resident matrices: gradient mix differs")
	}
	type binding struct {
		weight   *linWeight
		gradient driver.DevicePtr
	}
	bindings := []binding{
		{&weights.mlp.gate, gradients.MLPGate},
		{&weights.mlp.up, gradients.MLPUp},
		{&weights.mlp.down, gradients.MLPDown},
	}
	if rw.IsLinear {
		bindings = append(bindings,
			binding{&weights.gdn.wq, gradients.GDNWq}, binding{&weights.gdn.wk, gradients.GDNWk},
			binding{&weights.gdn.wv, gradients.GDNWv}, binding{&weights.gdn.wbeta, gradients.GDNWbeta},
			binding{&weights.gdn.walpha, gradients.GDNWalpha}, binding{&weights.gdn.wz, gradients.GDNWz},
			binding{&weights.gdn.wout, gradients.GDNWout})
	} else {
		bindings = append(bindings,
			binding{&weights.attn.wq, gradients.AttnWq}, binding{&weights.attn.wk, gradients.AttnWk},
			binding{&weights.attn.wv, gradients.AttnWv}, binding{&weights.attn.wo, gradients.AttnWo})
	}
	for index, target := range bindings {
		if target.weight.dev == 0 || target.gradient == 0 {
			return hybridMatW{}, fmt.Errorf("hybrid resident matrices: missing weight or gradient")
		}
		for other, source := range bindings {
			if target.gradient == source.weight.dev {
				return hybridMatW{}, fmt.Errorf("hybrid resident matrices: gradient aliases weight")
			}
			if index != other && target.gradient == source.gradient {
				return hybridMatW{}, fmt.Errorf("hybrid resident matrices: duplicate gradient view")
			}
		}
		target.weight.gradient = target.gradient
	}
	return weights, nil
}

// HybridDecoderLayerForwardDeviceResident is HybridDecoderLayerForwardDevice with
// the layer's MATRIX weights read from resident device pointers (rw) instead of
// uploaded host slices; the vector weights come from w. Composed from the exact
// same parity-verified ops -- it is the resident-weight arm of the one-owner
// hybridLayerForwardW. No matrix weight is uploaded or read back.
func HybridDecoderLayerForwardDeviceResident(worker *device.Worker, x []float32, rw HybridLayerResidentMatrices, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state []float32) ([]float32, HybridLayerDeviceCache, error) {
	return hybridLayerForwardW(worker, x, rw.matW(), w, d, state)
}

// HybridDecoderLayerBackwardDeviceResident is the resident-weight arm of
// hybridLayerBackwardW: matrix weights read from rw, vector weights from w. The
// cache must come from the corresponding forward with unchanged inputs, state
// and weights. Its host activations are consumed once and released. Matrix
// gradients overwrite the caller's disjoint gradient views and remain resident;
// their returned host slices are nil. Vector and activation gradients return
// to the host. Neither slab is owned or released by the operator.
func HybridDecoderLayerBackwardDeviceResident(worker *device.Worker, x []float32, rw, gradients HybridLayerResidentMatrices, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state, dOut []float32, cache *HybridLayerDeviceCache) (hostmath.HybridDecoderLayerGrads, error) {
	if rw.IsLinear != w.IsLinear {
		return hostmath.HybridDecoderLayerGrads{}, fmt.Errorf("hybrid resident matrices: weight mix differs")
	}
	mw, err := rw.withGradients(gradients)
	if err != nil {
		return hostmath.HybridDecoderLayerGrads{}, err
	}
	return hybridLayerBackwardCachedW(worker, x, mw, w, d, state, dOut, cache)
}
