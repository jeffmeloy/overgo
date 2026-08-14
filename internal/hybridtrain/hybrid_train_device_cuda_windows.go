//go:build windows

package hybridtrain

import (
	"overgo/internal/cuda/device"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
)

// residency reports, for the device loop, how the layer matrix weights and their
// Muon momentum are held across steps -- surfaced so the parity test can assert the
// residency contract (weights + momentum uploaded ONCE, neither round-tripped
// per step: densecausal-class ZERO per-step weight motion).
type residency struct {
	MatrixElems     int // resident matrix weight/grad/momentum buffer size
	WeightUploads   int // host->device matrix-weight uploads over the whole run (want 1)
	MomentumUploads int // host->device momentum uploads over the whole run (want 1)
	MomentumReads   int // device->host momentum reads over the whole run (want 0 until checkpoint)
	WeightReads     int // device->host matrix-weight reads PER STEP (want 0: weights resident)
	FinalWeightRead int // device->host matrix-weight reads at the final checkpoint (want 1)
	GradUploads     int // host->device grad uploads (one per step: grads are recomputed each step)
	Steps           int
}

// TrainDeviceResident runs K steps of the SAME hybrid stack + optimizer as TrainHost,
// but with the resident-set matrix weights, their gradients and their Muon momentum
// living in persistent device buffers: matrix weights upload ONCE (AllocResidentF32),
// the Muon Newton-Schulz update runs in place on those buffers every step
// (DeviceMuonMatricesResident), and NEITHER the weight buffer NOR the momentum buffer
// is round-tripped per step. The per-layer forward/backward now consume RESIDENT
// weight POINTERS (HybridDecoderLayerForwardDeviceResident /
// HybridDecoderLayerBackwardDeviceResident): each layer's matrix weights are addressed
// as sub-pointers of dW via ResidentPtr, so the updated weights are read straight from
// the device on the next step with NO device->host copy (WeightReads == 0). Only the
// freshly recomputed gradients upload into the resident grad buffer once per step
// (GradUploads) and the activations round-trip (inherent to the op-composition ops).
// The matrix weights are pulled back to host exactly ONCE, at the final checkpoint
// (FinalWeightRead), so m.matW/m.Weights reflect the trained model on return.
// The vector params (norms, GDN conv/bias/scalars) take the host Sign update, exactly
// as TrainHost, so any host<->device trajectory divergence is isolated to the device
// Muon matrices + the device fp32 forward/backward. Returns the loss trajectory and
// the residency accounting.
func (m *Model) TrainDeviceResident(worker *device.Worker, steps int, cfg optimizer.Config) ([]float64, residency, error) {
	acc := residency{MatrixElems: len(m.matW), Steps: steps}

	plans, err := m.matrixPlans()
	if err != nil {
		return nil, acc, err
	}
	if err := validateMatrixTiling(plans, len(m.matW)); err != nil {
		return nil, acc, err
	}

	// Persistent device buffers: matrix weights (uploaded once), gradients, momentum.
	dW, err := devicemath.AllocResidentF32(worker, len(m.matW), m.matW)
	if err != nil {
		return nil, acc, err
	}
	acc.WeightUploads++
	dG, err := devicemath.AllocResidentF32(worker, len(m.matW), nil)
	if err != nil {
		_ = devicemath.FreeResident(worker, dW)
		return nil, acc, err
	}
	dM, err := devicemath.AllocResidentF32(worker, len(m.matW), nil)
	if err != nil {
		_ = devicemath.FreeResident(worker, dW, dG)
		return nil, acc, err
	}
	acc.MomentumUploads++ // dM zero-initialised on the device, once
	defer func() { _ = devicemath.FreeResident(worker, dW, dG, dM) }()

	// One resident Muon target per matrix, addressing its sub-range of dW/dG/dM.
	updates := make([]optimizer.ResidentMatrix, len(m.mats))
	for i, md := range m.mats {
		updates[i] = optimizer.ResidentMatrix{
			Weights:  devicemath.ResidentPtr(dW, md.off),
			Gradient: devicemath.ResidentPtr(dG, md.off),
			Momentum: devicemath.ResidentPtr(dM, md.off),
			Rows:     md.rows, Cols: md.cols,
		}
	}

	// Per-layer RESIDENT weight-pointer bundles: each layer's matrix weights are
	// device pointers into dW (computed once), so the forward/backward read the
	// current weights in place -- no per-step host<->device weight motion.
	rw := make([]devicemath.HybridLayerResidentWeights, len(plans))
	for i, p := range plans {
		rw[i] = devicemath.HybridLayerResidentWeights{
			IsLinear: p.IsLinear,
			MLPGate:  devicemath.ResidentPtr(dW, p.Gate.Off),
			MLPUp:    devicemath.ResidentPtr(dW, p.Up.Off),
			MLPDown:  devicemath.ResidentPtr(dW, p.Down.Off),
		}
		if p.IsLinear {
			rw[i].GDNWq = devicemath.ResidentPtr(dW, p.GWq.Off)
			rw[i].GDNWk = devicemath.ResidentPtr(dW, p.GWk.Off)
			rw[i].GDNWv = devicemath.ResidentPtr(dW, p.GWv.Off)
			rw[i].GDNWbeta = devicemath.ResidentPtr(dW, p.GWbeta.Off)
			rw[i].GDNWalpha = devicemath.ResidentPtr(dW, p.GWalpha.Off)
			rw[i].GDNWz = devicemath.ResidentPtr(dW, p.GWz.Off)
			rw[i].GDNWout = devicemath.ResidentPtr(dW, p.GWout.Off)
		} else {
			rw[i].AttnWq = devicemath.ResidentPtr(dW, p.Wq.Off)
			rw[i].AttnWk = devicemath.ResidentPtr(dW, p.Wk.Off)
			rw[i].AttnWv = devicemath.ResidentPtr(dW, p.Wv.Off)
			rw[i].AttnWo = devicemath.ResidentPtr(dW, p.Wo.Off)
		}
	}

	// Vector params run on the host optimizer (Sign), same as TrainHost.
	vecGrad := make([]float32, len(m.vecW))
	vecOpt, err := optimizer.New(m.vecW, vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, acc, err
	}

	traj := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		// Forward stack on device: matrix weights read from resident dW pointers,
		// vector weights from the host slices (aliasing vecW).
		x := m.X
		inputs := make([][]float32, len(m.Weights))
		for i := range m.Weights {
			inputs[i] = x
			o, _, err := devicemath.HybridDecoderLayerForwardDeviceResident(worker, x, rw[i], m.Weights[i], m.Dims[i], m.States[i])
			if err != nil {
				return nil, acc, err
			}
			x = o
		}
		loss, dTop := m.loss(x)
		traj = append(traj, loss)

		// Backward stack on device (recomputes its own forward intermediates from
		// the resident weights).
		grads := make([]hostmath.HybridDecoderLayerGrads, len(m.Weights))
		dOut := dTop
		for i := len(m.Weights) - 1; i >= 0; i-- {
			g, err := devicemath.HybridDecoderLayerBackwardDeviceResident(worker, inputs[i], rw[i], m.Weights[i], m.Dims[i], m.States[i], dOut)
			if err != nil {
				return nil, acc, err
			}
			grads[i] = g
			dOut = g.DX
		}
		matG, vecG := m.packGrads(grads)

		// Refresh the resident gradient buffer (grads are recomputed each step).
		if err := devicemath.WriteResident(worker, dG, devicemath.ResidentSlice{ElemOffset: 0, Data: matG}); err != nil {
			return nil, acc, err
		}
		acc.GradUploads++

		// Host Sign update for vector params (aliases vecW -> next forward sees it).
		copy(vecGrad, vecG)
		vecOpt.Step()

		// Resident Muon: matrix weights + momentum updated in place on the device.
		// Nothing is uploaded or downloaded here; weights AND momentum stay resident.
		if err := optimizer.DeviceMuonMatricesResident(worker, updates, step+1, cfg); err != nil {
			return nil, acc, err
		}
		// NO per-step weight read-back: the next forward reads dW in place.
	}

	// Checkpoint: pull the trained matrix weights back to host exactly once, so
	// m.matW (and the aliasing m.Weights) reflect the trained model on return.
	if err := devicemath.ReadResident(worker, dW, devicemath.ResidentSlice{ElemOffset: 0, Data: m.matW}); err != nil {
		return nil, acc, err
	}
	acc.FinalWeightRead++
	// The loop never read dM, so MomentumReads stays 0.
	return traj, acc, nil
}
