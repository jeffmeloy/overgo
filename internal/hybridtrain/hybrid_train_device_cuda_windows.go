//go:build windows

package hybridtrain

import (
	"fmt"

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

type hostMasterTraining struct {
	hostBackwardPass
	worker   *device.Worker
	config   optimizer.Config
	momentum []float32
	vecOpt   *optimizer.Optimizer
}

func (training *hostMasterTraining) Step(step int) error {
	if err := optimizer.DeviceMuonStepPlanStreamed(
		training.worker, training.model.matW, training.matGrad, training.momentum,
		training.model.matPlan, step, training.config,
	); err != nil {
		return err
	}
	training.vecOpt.Step()
	return nil
}

// TrainHostMasterStreamed trains with host-resident f32 masters, momentum and
// gradients, streaming each Muon matrix group across the device per step.
// Peak device memory is bounded by the largest single matrix, never the
// parameter count, so the full stack of a multi-billion-parameter artifact
// trains on a device that cannot hold its weights. observe (optional) sees
// each committed step's loss and may stop training by returning an error.
func (m *Model) TrainHostMasterStreamed(worker *device.Worker, steps int, cfg optimizer.Config, observe func(step int, loss float64) error) ([]float64, error) {
	pass := newHostBackwardPass(m)
	vecOpt, err := optimizer.New(m.vecW, pass.vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, err
	}
	training := &hostMasterTraining{
		hostBackwardPass: pass,
		worker:           worker, config: cfg,
		momentum: make([]float32, len(m.matW)), vecOpt: vecOpt,
	}
	return runTrainingObserved(m.program, training, m.Target, steps, observe)
}

// hostMasterLayerStreamedTraining: forward on host f32 masters; the backward
// walk fuses each layer's streamed Muon step as that layer's gradients
// complete, so host gradient residency is bounded by ONE layer's reusable
// scratch instead of a full parameter-count slab. Momentum stays a full host
// f32 slab (it must persist across steps); the bounded triple is therefore
// masters + momentum + one-layer scratch — the lane for stacks whose full
// masters+gradients+momentum triplication exceeds host memory.
type hostMasterLayerStreamedTraining struct {
	model      *Model
	worker     *device.Worker
	config     optimizer.Config
	trace      hostmath.HybridStackTrace
	layers     []layerStreamPlan
	matScratch []float32 // one-layer gradient window, reused across layers and steps
	momentum   []float32 // full matrix momentum slab, persists across steps
	vecGrad    []float32
	vecOpt     *optimizer.Optimizer
	stepped    int // committed steps; the backward walk applies Muon step stepped+1
}

func (training *hostMasterLayerStreamedTraining) Forward() ([]float32, error) {
	model := training.model
	output, trace := hostmath.HybridStackForward(model.X, model.Weights, model.Dims, model.States)
	training.trace = trace
	return output, nil
}

func (training *hostMasterLayerStreamedTraining) Backward(dTop []float32) error {
	model := training.model
	_, err := hostmath.HybridStackBackwardStreamed(&training.trace, model.Weights, model.Dims, model.States, dTop,
		func(layer int, g hostmath.HybridDecoderLayerGrads) error {
			// Update ordering: this visitor runs AFTER the layer's DX and weight
			// gradients were computed from its PRE-step weights, so stepping the
			// layer here cannot perturb the remaining backward — inner layers
			// consume only DX and their own saved activations. Stepping before
			// the layer's own backward would be wrong; stepping after it is the
			// full-slab trajectory exactly.
			lp := training.layers[layer]
			mat := training.matScratch[:lp.matSize]
			vec := training.vecGrad[lp.vecOff : lp.vecOff+lp.vecSize]
			if mi, vi := packLayerGradients(g, mat, vec); mi != lp.matSize || vi != lp.vecSize {
				return fmt.Errorf("layer %d packed %d+%d gradient values, window is %d+%d", layer, mi, vi, lp.matSize, lp.vecSize)
			}
			return optimizer.DeviceMuonStepPlanStreamed(
				training.worker,
				model.matW[lp.matOff:lp.matOff+lp.matSize], mat,
				training.momentum[lp.matOff:lp.matOff+lp.matSize],
				lp.plan, training.stepped+1, training.config,
			)
		})
	return err
}

func (training *hostMasterLayerStreamedTraining) Step(step int) error {
	if step != training.stepped+1 {
		return fmt.Errorf("layer-streamed step %d does not follow committed step %d", step, training.stepped)
	}
	// Matrix groups were stepped during the backward walk; the vector step
	// commits here so the optimize phase still closes every step.
	training.vecOpt.Step()
	training.stepped = step
	return nil
}

// TrainHostMasterLayerStreamed trains with host-resident f32 masters and
// momentum while gradients NEVER triplicate: each layer's gradients land in a
// reusable one-layer scratch during the backward walk and that layer's Muon
// step (streamed per matrix group across the device) applies immediately.
// Peak host memory is masters + momentum + one layer of scratch; peak device
// memory stays bounded by the largest single matrix. The Muon math is the
// full-slab lane's unchanged — per-group f32 momentum and device
// Newton-Schulz — so the trajectory matches TrainHostMasterStreamed exactly.
func (m *Model) TrainHostMasterLayerStreamed(worker *device.Worker, steps int, cfg optimizer.Config, observe func(step int, loss float64) error) ([]float64, error) {
	plans, err := m.layerStreamPlans()
	if err != nil {
		return nil, err
	}
	maxLayer := 0
	for _, p := range plans {
		maxLayer = max(maxLayer, p.matSize)
	}
	vecGrad := make([]float32, len(m.vecW))
	vecOpt, err := optimizer.New(m.vecW, vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, err
	}
	training := &hostMasterLayerStreamedTraining{
		model: m, worker: worker, config: cfg,
		layers:     plans,
		matScratch: make([]float32, maxLayer),
		momentum:   make([]float32, len(m.matW)),
		vecGrad:    vecGrad, vecOpt: vecOpt,
	}
	return runTrainingObserved(m.program, training, m.Target, steps, observe)
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
