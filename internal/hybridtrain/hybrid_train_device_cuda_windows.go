//go:build windows

package hybridtrain

import (
	"overgo/internal/cuda/device"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
)

// residency records device-loop weight and momentum movement.
type residency struct {
	MatrixElems     int // resident weight/gradient/momentum elements
	WeightUploads   int // one initial upload
	MomentumUploads int // one initial upload
	MomentumReads   int // zero before checkpoint
	WeightReads     int // zero per step
	FinalWeightRead int // one checkpoint read
	GradUploads     int // one per step
	Steps           int
}

// TrainDeviceResident runs hybrid forward/backward with resident matrix weights,
// gradients, and Muon momentum. Matrix state crosses the host boundary only at
// initialization and final checkpoint. Vector parameters retain the host Sign path.
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
	residentMuon, err := optimizer.NewResidentMatrixMuonPlan(worker, updates, cfg)
	if err != nil {
		return nil, acc, err
	}
	defer residentMuon.Close()

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
		if err := residentMuon.StepMatrices(updates, step+1); err != nil {
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
