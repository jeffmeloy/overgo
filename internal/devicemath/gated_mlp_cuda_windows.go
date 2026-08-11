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
	if rows <= 0 || d <= 0 || inter <= 0 ||
		len(x) != rows*d || len(wGate) != d*inter || len(wUp) != d*inter || len(wDown) != inter*d ||
		len(g) != rows*inter || len(a) != rows*inter || len(u) != rows*inter || len(h) != rows*inter || len(dY) != rows*d {
		return GatedMLPGrads{}, fmt.Errorf("GatedMLPBackward: shape mismatch (rows=%d d=%d inter=%d)", rows, d, inter)
	}

	// Y = h·Wdown  ->  dh[rows,inter], dWdown[inter,d].
	dh, dWDown, err := LinearBackward(worker, h, wDown, dY, rows, inter, d)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	// h = a⊙u  ->  da = dh⊙u ; du = dh⊙a.
	da := make([]float32, rows*inter)
	du := make([]float32, rows*inter)
	for i := range dh {
		da[i] = dh[i] * u[i]
		du[i] = dh[i] * a[i]
	}
	// a = silu(g)  ->  dg = da⊙silu'(g).
	dg, err := SiLUBackward(worker, g, da)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	// g = X·Wgate  ->  dXgate[rows,d], dWgate[d,inter].
	dXGate, dWGate, err := LinearBackward(worker, x, wGate, dg, rows, d, inter)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	// u = X·Wup  ->  dXup[rows,d], dWup[d,inter].
	dXUp, dWUp, err := LinearBackward(worker, x, wUp, du, rows, d, inter)
	if err != nil {
		return GatedMLPGrads{}, err
	}
	// X feeds both gate and up: dX = dXgate + dXup.
	dX := make([]float32, rows*d)
	for i := range dX {
		dX[i] = dXGate[i] + dXUp[i]
	}
	return GatedMLPGrads{DX: dX, DWGate: dWGate, DWUp: dWUp, DWDown: dWDown}, nil
}
