//go:build windows

package hybridtrain

import (
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
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

type residentTraining struct {
	model            *Model
	worker           *device.Worker
	weights          []devicemath.HybridLayerResidentWeights
	inputs           [][]float32
	grads            []hostmath.HybridDecoderLayerGrads
	matGrad, vecGrad []float32
	dGradient        driver.DevicePtr
	updates          []optimizer.ResidentMatrix
	matOpt           *optimizer.ResidentMuonPlan
	vecOpt           *optimizer.Optimizer
	residency        *residency
}

func (training *residentTraining) Forward() ([]float32, error) {
	model := training.model
	if len(training.inputs) != len(model.Weights) {
		training.inputs = make([][]float32, len(model.Weights))
	}
	x := model.X
	for index := range model.Weights {
		training.inputs[index] = x
		output, _, err := devicemath.HybridDecoderLayerForwardDeviceResident(
			training.worker, x, training.weights[index], model.Weights[index], model.Dims[index], model.States[index],
		)
		if err != nil {
			return nil, err
		}
		x = output
	}
	return x, nil
}

func (training *residentTraining) Backward(dTop []float32) error {
	model := training.model
	if len(training.grads) != len(model.Weights) {
		training.grads = make([]hostmath.HybridDecoderLayerGrads, len(model.Weights))
	}
	dOut := dTop
	for index := len(model.Weights) - 1; index >= 0; index-- {
		gradient, err := devicemath.HybridDecoderLayerBackwardDeviceResident(
			training.worker, training.inputs[index], training.weights[index], model.Weights[index], model.Dims[index], model.States[index], dOut,
		)
		if err != nil {
			return err
		}
		training.grads[index] = gradient
		dOut = gradient.DX
	}
	model.packGradients(training.grads, training.matGrad, training.vecGrad)
	return nil
}

func (training *residentTraining) Step(step int) error {
	if err := devicemath.WriteResident(training.worker, training.dGradient, devicemath.ResidentSlice{Data: training.matGrad}); err != nil {
		return err
	}
	training.residency.GradUploads++
	if err := training.matOpt.StepMatrices(training.updates, step); err != nil {
		return err
	}
	training.vecOpt.Step()
	return nil
}

// TrainDeviceResident keeps matrix state resident through the compiled loop.
func (m *Model) TrainDeviceResident(worker *device.Worker, steps int, cfg optimizer.Config) ([]float64, residency, error) {
	acc := residency{MatrixElems: len(m.matW), Steps: steps}

	plans, err := m.matrixPlans()
	if err != nil {
		return nil, acc, err
	}
	if err := validateMatrixTiling(plans, len(m.matW)); err != nil {
		return nil, acc, err
	}

	// Session-resident matrix state.
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

	// Matrix update views.
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

	// Per-layer resident views.
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

	vecGrad := make([]float32, len(m.vecW))
	vecOpt, err := optimizer.New(m.vecW, vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, acc, err
	}
	training := &residentTraining{
		model: m, worker: worker, weights: rw,
		matGrad: make([]float32, len(m.matW)), vecGrad: vecGrad,
		dGradient: dG, updates: updates, matOpt: residentMuon, vecOpt: vecOpt, residency: &acc,
	}
	trajectory, err := runTraining(m.program, training, m.Target, steps)
	if err != nil {
		return nil, acc, err
	}

	// Final checkpoint readback.
	if err := devicemath.ReadResident(worker, dW, devicemath.ResidentSlice{ElemOffset: 0, Data: m.matW}); err != nil {
		return nil, acc, err
	}
	acc.FinalWeightRead++
	return trajectory, acc, nil
}
