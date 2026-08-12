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
// residency contract (weights + momentum uploaded ONCE, momentum never round-tripped).
type residency struct {
	MatrixElems     int // resident matrix weight/grad/momentum buffer size
	WeightUploads   int // host->device matrix-weight uploads over the whole run (want 1)
	MomentumUploads int // host->device momentum uploads over the whole run (want 1)
	MomentumReads   int // device->host momentum reads over the whole run (want 0 until checkpoint)
	WeightReads     int // device->host matrix-weight reads (one per step: host-fed fwd needs them)
	GradUploads     int // host->device grad uploads (one per step: grads are recomputed each step)
	Steps           int
}

// TrainDeviceResident runs K steps of the SAME hybrid stack + optimizer as TrainHost,
// but with the resident-set matrix weights, their gradients and their Muon momentum
// living in persistent device buffers: matrix weights upload ONCE (AllocResidentF32),
// the Muon Newton-Schulz update runs in place on those buffers every step
// (DeviceMuonMatricesResident), and the momentum buffer is never round-tripped until
// the final checkpoint. The per-layer forward/backward use the parity-verified
// HybridDecoderLayerForwardDevice / HybridDecoderLayerBackwardDevice; because those
// ops are host-weight-fed, the updated matrix weights are read back to host once per
// step (WeightReads) so the next forward sees them, and the freshly recomputed
// gradients upload into the resident grad buffer once per step (GradUploads) -- but
// the weight and momentum buffers themselves are allocated/uploaded exactly once.
// The vector params (norms, GDN conv/bias/scalars) take the host Sign update, exactly
// as TrainHost, so any host<->device trajectory divergence is isolated to the device
// Muon matrices + the device fp32 forward/backward. Returns the loss trajectory and
// the residency accounting.
func (m *Model) TrainDeviceResident(worker *device.Worker, steps int, cfg optimizer.Config) ([]float64, residency, error) {
	acc := residency{MatrixElems: len(m.matW), Steps: steps}

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

	// Vector params run on the host optimizer (Sign), same as TrainHost.
	vecGrad := make([]float32, len(m.vecW))
	vecOpt, err := optimizer.New(m.vecW, vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, acc, err
	}

	traj := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		// Forward stack on device (weights read from the host slices aliasing matW).
		x := m.X
		inputs := make([][]float32, len(m.Weights))
		for i := range m.Weights {
			inputs[i] = x
			o, _, err := devicemath.HybridDecoderLayerForwardDevice(worker, x, m.Weights[i], m.Dims[i], m.States[i])
			if err != nil {
				return nil, acc, err
			}
			x = o
		}
		loss, dTop := m.loss(x)
		traj = append(traj, loss)

		// Backward stack on device (recomputes its own forward intermediates).
		grads := make([]hostmath.HybridDecoderLayerGrads, len(m.Weights))
		dOut := dTop
		for i := len(m.Weights) - 1; i >= 0; i-- {
			g, err := devicemath.HybridDecoderLayerBackwardDevice(worker, inputs[i], m.Weights[i], m.Dims[i], m.States[i], dOut)
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
		// Nothing is uploaded or downloaded here; momentum stays resident in dM.
		if err := optimizer.DeviceMuonMatricesResident(worker, updates, step+1, cfg); err != nil {
			return nil, acc, err
		}

		// Read the updated matrix weights back so the (host-fed) next forward sees
		// them. This is the ONLY per-step matrix-weight transfer; momentum stays put.
		if err := devicemath.ReadResident(worker, dW, devicemath.ResidentSlice{ElemOffset: 0, Data: m.matW}); err != nil {
			return nil, acc, err
		}
		acc.WeightReads++
	}
	// Checkpoint would read dM here; the loop never did, so MomentumReads stays 0.
	return traj, acc, nil
}
