//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
)

// GatedMLPGrads holds the gradients of a SwiGLU/gated-MLP block.
type GatedMLPGrads struct {
	DX     []float32 // [rows, d]
	DWGate []float32 // [d, inter]
	DWUp   []float32 // [d, inter]
	DWDown []float32 // [inter, d]
}

// GatedMLPBackward computes the gradients of the SwiGLU block
//
//	g = X·Wgate ; a = silu(g) ; u = X·Wup ; h = a⊙u ; Y = h·Wdown
//
// given the output cotangent dY[rows,d] and the forward's saved activations
// g, a, u, h (all [rows,inter]). It composes the device LinearBackward and
// SiLUBackward operators with host elementwise glue -- no new kernel. This is
// the MLP-half of a transformer layer's device backward. Correctness-first: each
// sub-op takes its own device context; a fused resident version is a later
// optimization.
func GatedMLPBackward(worker *device.Worker, x, wGate, wUp, wDown, g, a, u, h, dY []float32, rows, d, inter int) (GatedMLPGrads, error) {
	return gatedMLPBackward(worker, "GatedMLPBackward", LinearBackward, x, wGate, wUp, wDown, g, a, u, h, dY, rows, d, inter)
}

// GatedMLPBackwardT is GatedMLPBackward for densecausal's HF weight layout, where
// every projection is Y = X·Wᵀ with W stored [out,in] (gate/up are [inter,d],
// down is [d,inter]). Uses LinearBackwardT for the three projections; the
// activation glue is identical. Returns dX[rows,d] and the three weight grads in
// [out,in] layout.
func GatedMLPBackwardT(worker *device.Worker, x, wGate, wUp, wDown, g, a, u, h, dY []float32, rows, d, inter int) (GatedMLPGrads, error) {
	return gatedMLPBackward(worker, "GatedMLPBackwardT", LinearBackwardT, x, wGate, wUp, wDown, g, a, u, h, dY, rows, d, inter)
}

type linearBackwardOperator func(
	worker *device.Worker,
	x, weight, gradient []float32,
	rows, in, out int,
) (inputGradient, weightGradient []float32, err error)

func gatedMLPBackward(
	worker *device.Worker,
	operator string,
	linear linearBackwardOperator,
	x, wGate, wUp, wDown, g, a, u, h, dY []float32,
	rows, d, inter int,
) (GatedMLPGrads, error) {
	if rows <= 0 || d <= 0 || inter <= 0 ||
		len(x) != rows*d || len(wGate) != inter*d || len(wUp) != inter*d || len(wDown) != d*inter ||
		len(g) != rows*inter || len(a) != rows*inter || len(u) != rows*inter || len(h) != rows*inter || len(dY) != rows*d {
		return GatedMLPGrads{}, fmt.Errorf("%s: shape mismatch (rows=%d d=%d inter=%d)", operator, rows, d, inter)
	}
	dh, dWDown, err := linear(worker, h, wDown, dY, rows, inter, d)
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
	dXGate, dWGate, err := linear(worker, x, wGate, dg, rows, d, inter)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	dXUp, dWUp, err := linear(worker, x, wUp, du, rows, d, inter)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	dX := make([]float32, rows*d)
	for i := range dX {
		dX[i] = dXGate[i] + dXUp[i]
	}
	return GatedMLPGrads{DX: dX, DWGate: dWGate, DWUp: dWUp, DWDown: dWDown}, nil
}
