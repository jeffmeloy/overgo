//go:build windows

package devicemath

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// mlpMatW binds the three projection matrices and optional gradient destinations.
type mlpMatW struct{ gate, up, down linWeight }

// GatedMLPBackwardTResident is the resident counterpart to GatedMLPBackwardT: it
// runs the entire densecausal-convention SwiGLU MLP backward inside ONE
// worker.Do, with a single cuBLAS handle and ops_f32 module, uploading the
// inputs once and keeping every intermediate gradient (dh, da, du, dg, dXgate,
// dXup) resident on the device -- no per-op host round-trips. Only dX and the
// three weight grads come back. Same math as GatedMLPBackwardT; this addresses
// SQA finding 2/3 (residency) for the MLP block.
func GatedMLPBackwardTResident(worker *device.Worker, x, wGate, wUp, wDown, g, a, u, h, dY []float32, rows, d, inter int) (GatedMLPGrads, error) {
	return gatedMLPBackwardTW(worker, x, mlpMatW{hostW(wGate), hostW(wUp), hostW(wDown)}, g, a, u, h, dY, rows, d, inter)
}

// gatedMLPBackwardTW shares the resident SwiGLU VJP for host and borrowed
// matrix bindings. Intermediate gradients stay in this session; optional
// caller-owned matrix gradients remain resident after it closes.
func gatedMLPBackwardTW(worker *device.Worker, x []float32, mw mlpMatW, g, a, u, h, dY []float32, rows, d, inter int) (GatedMLPGrads, error) {
	if rows <= 0 || d <= 0 || inter <= 0 ||
		len(x) != rows*d ||
		len(g) != rows*inter || len(a) != rows*inter || len(u) != rows*inter || len(h) != rows*inter || len(dY) != rows*d {
		return GatedMLPGrads{}, fmt.Errorf("GatedMLPBackwardTResident: shape mismatch (rows=%d d=%d inter=%d)", rows, d, inter)
	}
	bindings := []linWeight{mw.gate, mw.up, mw.down}
	for index, binding := range bindings {
		if binding.host == nil && binding.dev == 0 || binding.host != nil && len(binding.host) != d*inter {
			return GatedMLPGrads{}, fmt.Errorf("resident MLP: missing or malformed matrix %d", index)
		}
		if binding.gradient != 0 {
			for other, source := range bindings {
				if binding.gradient == source.dev || index != other && binding.gradient == source.gradient {
					return GatedMLPGrads{}, fmt.Errorf("resident MLP: aliased gradient %d", index)
				}
			}
		}
	}
	dX := make([]float32, rows*d)
	var dWGate, dWUp, dWDown []float32

	err := withCUDABLAS(worker, func(session *cudaBLAS) error {
		mulFn, err := session.function("multiply_f32")
		if err != nil {
			return err
		}
		addFn, err := session.function("add_f32")
		if err != nil {
			return err
		}
		siluFn, err := session.function("silu_backward_f32")
		if err != nil {
			return err
		}

		// Resident inputs (uploaded once).
		hnP, err := session.upload(x)
		if err != nil {
			return err
		}
		wGateP, err := mw.gate.resolve(session, inter*d)
		if err != nil {
			return err
		}
		wUpP, err := mw.up.resolve(session, inter*d)
		if err != nil {
			return err
		}
		wDownP, err := mw.down.resolve(session, d*inter)
		if err != nil {
			return err
		}
		gP, err := session.upload(g)
		if err != nil {
			return err
		}
		aP, err := session.upload(a)
		if err != nil {
			return err
		}
		uP, err := session.upload(u)
		if err != nil {
			return err
		}
		hP, err := session.upload(h)
		if err != nil {
			return err
		}
		dyP, err := session.upload(dY)
		if err != nil {
			return err
		}
		// Resident intermediates + outputs.
		dhP, err := session.alloc(rows * inter)
		if err != nil {
			return err
		}
		daP, err := session.alloc(rows * inter)
		if err != nil {
			return err
		}
		duP, err := session.alloc(rows * inter)
		if err != nil {
			return err
		}
		dgP, err := session.alloc(rows * inter)
		if err != nil {
			return err
		}
		dxGateP, err := session.alloc(rows * d)
		if err != nil {
			return err
		}
		dxUpP, err := session.alloc(rows * d)
		if err != nil {
			return err
		}
		dxP, err := session.alloc(rows * d)
		if err != nil {
			return err
		}
		dwGateP, hostGate, err := mw.gate.allocateGradient(session, inter*d)
		if err != nil {
			return err
		}
		dwUpP, hostUp, err := mw.up.allocateGradient(session, inter*d)
		if err != nil {
			return err
		}
		dwDownP, hostDown, err := mw.down.allocateGradient(session, d*inter)
		if err != nil {
			return err
		}
		dWGate, dWUp, dWDown = hostGate, hostUp, hostDown

		mul := func(x, y, out driver.DevicePtr, n int) error {
			return session.launchVector3(mulFn, x, y, out, n)
		}
		add := func(x, y, out driver.DevicePtr, n int) error {
			return session.launchVector3(addFn, x, y, out, n)
		}
		// silu_backward_f32 arg order is (dy, x, out): out = dy * silu'(x).
		silu := func(x, dy, out driver.DevicePtr, n int) error {
			return session.launchVector3(siluFn, dy, x, out, n)
		}

		// Y = h·Wdownᵀ  ->  dh = dY·Wdown ; dWdown = dYᵀ·h.
		if err := session.gemm(false, false, rows, d, inter, dyP, wDownP, dhP); err != nil {
			return err
		}
		if err := session.gemm(true, false, d, rows, inter, dyP, hP, dwDownP); err != nil {
			return err
		}
		// da = dh⊙u ; du = dh⊙a ; dg = silu'(g)⊙da.
		if err := mul(dhP, uP, daP, rows*inter); err != nil {
			return err
		}
		if err := mul(dhP, aP, duP, rows*inter); err != nil {
			return err
		}
		if err := silu(gP, daP, dgP, rows*inter); err != nil {
			return err
		}
		// g = X·Wgateᵀ ->  dXgate = dg·Wgate ; dWgate = dgᵀ·X.
		if err := session.gemm(false, false, rows, inter, d, dgP, wGateP, dxGateP); err != nil {
			return err
		}
		if err := session.gemm(true, false, inter, rows, d, dgP, hnP, dwGateP); err != nil {
			return err
		}
		// u = X·Wupᵀ ->  dXup = du·Wup ; dWup = duᵀ·X.
		if err := session.gemm(false, false, rows, inter, d, duP, wUpP, dxUpP); err != nil {
			return err
		}
		if err := session.gemm(true, false, inter, rows, d, duP, hnP, dwUpP); err != nil {
			return err
		}
		// dX = dXgate + dXup.
		if err := add(dxGateP, dxUpP, dxP, rows*d); err != nil {
			return err
		}

		return session.finish(
			cudaDownload{dX, dxP}, cudaDownload{dWGate, dwGateP},
			cudaDownload{dWUp, dwUpP}, cudaDownload{dWDown, dwDownP},
		)
	})
	if err != nil {
		return GatedMLPGrads{}, err
	}
	return GatedMLPGrads{DX: dX, DWGate: dWGate, DWUp: dWUp, DWDown: dWDown}, nil
}
