//go:build windows

package devicemath

import (
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/hostmath"
)

// mlpMatW / attnMatW / gdnMatW bundle one layer's SwiGLU-MLP and active-mix
// projection matrices as linWeights -- either host slices (upload-per-call) or
// resident device pointers (uploaded once, read in place). hybridMatW is the
// per-layer union the layer forward/backward thread; the mix branch used is
// selected by w.IsLinear, so only the matching bundle is populated.
type mlpMatW struct{ gate, up, down linWeight }
type attnMatW struct{ wq, wk, wv, wo linWeight }
type gdnMatW struct{ wq, wk, wv, wbeta, walpha, wz, wout linWeight }

type hybridMatW struct {
	mlp  mlpMatW
	attn attnMatW
	gdn  gdnMatW
}

// hybridHostMatW / attnHostMatW / gdnHostMatW wrap host weight slices -- the
// legacy upload-per-call path the public host-weight-fed device ops use.
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

// HybridLayerResidentWeights carries one hybrid layer's MATRIX weights as device
// pointers into a persistent resident buffer (dW), computed once by the caller
// (ResidentPtr(dW, off) per matrix). The vector weights (norms, GDN conv/bias/
// scalars) are NOT here: they take the host Sign update and are passed as the
// host hostmath.HybridLayerWeights alongside. Populate the mix set matching
// IsLinear; the other set is ignored.
type HybridLayerResidentWeights struct {
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
func (rw HybridLayerResidentWeights) matW() hybridMatW {
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

// HybridDecoderLayerForwardDeviceResident is HybridDecoderLayerForwardDevice with
// the layer's MATRIX weights read from resident device pointers (rw) instead of
// uploaded host slices; the vector weights come from w. Composed from the exact
// same parity-verified ops -- it is the resident-weight arm of the one-owner
// hybridLayerForwardW. No matrix weight is uploaded or read back.
func HybridDecoderLayerForwardDeviceResident(worker *device.Worker, x []float32, rw HybridLayerResidentWeights, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state []float32) ([]float32, HybridLayerDeviceCache, error) {
	return hybridLayerForwardW(worker, x, rw.matW(), w, d, state)
}

// HybridDecoderLayerBackwardDeviceResident is the resident-weight arm of
// hybridLayerBackwardW: matrix weights read from rw, vector weights from w. The
// GDN residual mixOut is recomputed on device from rw, so it reads the CURRENT
// resident weights (the host slices are no longer refreshed per step). Weight
// gradients return as host slices for the caller to pack/upload; no matrix weight
// is read back.
func HybridDecoderLayerBackwardDeviceResident(worker *device.Worker, x []float32, rw HybridLayerResidentWeights, w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims, state, dOut []float32) (hostmath.HybridDecoderLayerGrads, error) {
	return hybridLayerBackwardW(worker, x, rw.matW(), w, d, state, dOut)
}

// gatedMLPBackwardTW is GatedMLPBackwardT over resident-or-host matrix weights
// (mw), same SwiGLU VJP composition (linearBackwardTW + SiLUBackward + host glue).
func gatedMLPBackwardTW(worker *device.Worker, x []float32, mw mlpMatW, g, a, u, h, dY []float32, rows, d, inter int) (GatedMLPGrads, error) {
	dh, dWDown, err := linearBackwardTW(worker, h, mw.down, dY, rows, inter, d)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	da := make([]float32, rows*inter)
	du := make([]float32, rows*inter)
	for i := range dh {
		da[i] = dh[i] * u[i]
		du[i] = dh[i] * a[i]
	}
	dg, err := SiLUBackward(worker, g, da)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	dXGate, dWGate, err := linearBackwardTW(worker, x, mw.gate, dg, rows, d, inter)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	dXUp, dWUp, err := linearBackwardTW(worker, x, mw.up, du, rows, d, inter)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	dX := make([]float32, rows*d)
	for i := range dX {
		dX[i] = dXGate[i] + dXUp[i]
	}
	return GatedMLPGrads{DX: dX, DWGate: dWGate, DWUp: dWUp, DWDown: dWDown}, nil
}
