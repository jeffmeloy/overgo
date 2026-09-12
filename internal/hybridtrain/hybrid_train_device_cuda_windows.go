//go:build windows

package hybridtrain

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/tensor/dtype"
)

// residency records device-loop weight and momentum movement.
type residency struct {
	MatrixElems     int // resident weight/gradient/momentum elements
	WeightUploads   int // one initial upload
	MomentumUploads int // one initial upload
	MomentumReads   int // zero before checkpoint
	WeightReads     int // zero per step
	FinalWeightRead int // one checkpoint read
	GradUploads     int // zero: matrix VJPs write directly into the resident slab
	Steps           int
}

type residentTraining struct {
	model     *Model
	worker    *device.Worker
	weights   []devicemath.HybridLayerResidentMatrices
	gradients []devicemath.HybridLayerResidentMatrices
	inputs    [][]float32
	caches    []devicemath.HybridLayerDeviceCache
	grads     []hostmath.HybridDecoderLayerGrads
	vecGrad   []float32
	updates   []optimizer.ResidentMatrix
	matOpt    *optimizer.ResidentMuonPlan
	vecOpt    *optimizer.Optimizer
}

func (training *residentTraining) Forward() ([]float32, error) {
	model := training.model
	if len(training.inputs) != len(model.Weights) {
		training.inputs = make([][]float32, len(model.Weights))
		training.caches = make([]devicemath.HybridLayerDeviceCache, len(model.Weights))
	}
	x := model.X
	for index := range model.Weights {
		training.inputs[index] = x
		output, cache, err := devicemath.HybridDecoderLayerForwardDeviceResident(
			training.worker, x, training.weights[index], model.Weights[index], model.Dims[index], model.States[index],
		)
		if err != nil {
			clear(training.caches)
			clear(training.inputs)
			return nil, err
		}
		training.caches[index] = cache
		x = output
	}
	return x, nil
}

func (training *residentTraining) Backward(dTop []float32) error {
	defer clear(training.caches)
	defer clear(training.inputs)
	defer func() { clear(training.grads) }()
	model := training.model
	if len(training.grads) != len(model.Weights) {
		training.grads = make([]hostmath.HybridDecoderLayerGrads, len(model.Weights))
	}
	dOut := dTop
	for index := len(model.Weights) - 1; index >= 0; index-- {
		gradient, err := devicemath.HybridDecoderLayerBackwardDeviceResident(
			training.worker, training.inputs[index], training.weights[index], training.gradients[index], model.Weights[index], model.Dims[index], model.States[index], dOut, &training.caches[index],
		)
		if err != nil {
			return err
		}
		training.grads[index] = gradient
		dOut = gradient.DX
	}
	model.packGradients(training.grads, nil, training.vecGrad)
	return nil
}

func (training *residentTraining) Step(step int) error {
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
// trains on a device that cannot hold its weights. options.Observe sees
// each committed step's loss and may stop training by returning an error.
func (m *Model) TrainHostMasterStreamed(worker *device.Worker, steps int, cfg optimizer.Config, options TrainingOptions) ([]float64, error) {
	mat, vec, err := m.resumeParts(steps, cfg, options, true)
	if err != nil {
		return nil, err
	}
	pass := newHostBackwardPass(m)
	vecOpt, err := optimizer.New(m.vecW, pass.vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, err
	}
	if err := restoreOptimizer(vecOpt, vec); err != nil {
		return nil, err
	}
	training := &hostMasterTraining{
		hostBackwardPass: pass,
		worker:           worker, config: cfg,
		momentum: make([]float32, len(m.matW)), vecOpt: vecOpt,
	}
	for index, value := range mat.Momentum {
		training.momentum[index] = float32(value)
	}
	trajectory, err := runTrainingObserved(m.program, training, m.Target, steps, mat.Step, options.Observe)
	if trajectory != nil && options.Checkpoint != nil {
		err = errors.Join(err, m.checkpointF32(options, training.momentum, mat.Step+len(trajectory), cfg, vecOpt))
	}
	return trajectory, err
}

// hostMasterLayerStreamedTraining: forward on host f32 masters; the backward
// walk fuses each layer's streamed Muon step as that layer's gradients
// complete, so host gradient residency is bounded by ONE layer's reusable
// scratch instead of a full parameter-count slab. Momentum stays a full host
// f32 slab (it must persist across steps); the bounded triple is therefore
// masters + momentum + one-layer scratch — the lane for stacks whose full
// masters+gradients+momentum triplication exceeds host memory.
type hostMasterLayerStreamedTraining struct {
	hostForwardPass
	worker     *device.Worker
	config     optimizer.Config
	layers     []layerStreamPlan
	matScratch []float32 // one-layer gradient window, reused across layers and steps
	momentum   []float32 // full matrix momentum slab, persists across steps
	vecGrad    []float32
	vecOpt     *optimizer.Optimizer
	stepped    int // committed steps; the backward walk applies Muon step stepped+1
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
func (m *Model) TrainHostMasterLayerStreamed(worker *device.Worker, steps int, cfg optimizer.Config, options TrainingOptions) ([]float64, error) {
	mat, vec, err := m.resumeParts(steps, cfg, options, true)
	if err != nil {
		return nil, err
	}
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
	if err := restoreOptimizer(vecOpt, vec); err != nil {
		return nil, err
	}
	training := &hostMasterLayerStreamedTraining{
		hostForwardPass: hostForwardPass{model: m},
		worker:          worker, config: cfg,
		layers:     plans,
		matScratch: make([]float32, maxLayer),
		momentum:   make([]float32, len(m.matW)),
		vecGrad:    vecGrad, vecOpt: vecOpt,
	}
	training.stepped = mat.Step
	for index, value := range mat.Momentum {
		training.momentum[index] = float32(value)
	}
	trajectory, err := runTrainingObserved(m.program, training, m.Target, steps, mat.Step, options.Observe)
	if trajectory != nil && options.Checkpoint != nil {
		err = errors.Join(err, m.checkpointF32(options, training.momentum, training.stepped, cfg, vecOpt))
	}
	return trajectory, err
}

// TrainDeviceResident keeps matrix state resident through the compiled loop.
func (m *Model) TrainDeviceResident(worker *device.Worker, steps int, cfg optimizer.Config, options TrainingOptions) ([]float64, residency, error) {
	acc := residency{MatrixElems: len(m.matW)}
	mat, vec, err := m.resumeParts(steps, cfg, options, true)
	if err != nil {
		return nil, acc, err
	}

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
	var initialMomentum []float32
	if options.Resume != nil {
		initialMomentum = dtype.Float64SliceToFloat32(mat.Momentum)
	}
	dM, err := devicemath.AllocResidentF32(worker, len(m.matW), initialMomentum)
	if err != nil {
		_ = devicemath.FreeResident(worker, dW, dG)
		return nil, acc, err
	}
	acc.MomentumUploads++ // one initial state upload or device zero initialization
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

	// Weight and gradient views share the validated optimizer layout.
	weights := make([]devicemath.HybridLayerResidentMatrices, len(plans))
	gradients := make([]devicemath.HybridLayerResidentMatrices, len(plans))
	for i, plan := range plans {
		weights[i] = plan.residentViews(dW)
		gradients[i] = plan.residentViews(dG)
	}
	vecGrad := make([]float32, len(m.vecW))
	vecOpt, err := optimizer.New(m.vecW, vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, acc, err
	}
	if err := restoreOptimizer(vecOpt, vec); err != nil {
		return nil, acc, err
	}
	training := &residentTraining{
		model: m, worker: worker, weights: weights, gradients: gradients,
		vecGrad: vecGrad, updates: updates, matOpt: residentMuon, vecOpt: vecOpt,
	}
	trajectory, err := runTrainingObserved(m.program, training, m.Target, steps, mat.Step, options.Observe)
	if trajectory == nil {
		return nil, acc, err
	}
	acc.Steps = len(trajectory)

	// Final checkpoint readback.
	if readErr := devicemath.ReadResident(worker, dW, devicemath.ResidentSlice{ElemOffset: 0, Data: m.matW}); readErr != nil {
		return trajectory, acc, errors.Join(err, readErr)
	}
	acc.FinalWeightRead++
	if options.Checkpoint != nil {
		momentum := make([]float32, len(m.matW))
		if readErr := devicemath.ReadResident(worker, dM, devicemath.ResidentSlice{Data: momentum}); readErr != nil {
			return trajectory, acc, errors.Join(err, readErr)
		}
		acc.MomentumReads++
		err = errors.Join(err, m.checkpointF32(options, momentum, mat.Step+len(trajectory), cfg, vecOpt))
	}
	return trajectory, acc, err
}

func (p layerMatrixPlan) residentViews(base driver.DevicePtr) devicemath.HybridLayerResidentMatrices {
	views := devicemath.HybridLayerResidentMatrices{
		IsLinear: p.IsLinear,
		MLPGate:  devicemath.ResidentPtr(base, p.Gate.Off),
		MLPUp:    devicemath.ResidentPtr(base, p.Up.Off),
		MLPDown:  devicemath.ResidentPtr(base, p.Down.Off),
	}
	if p.IsLinear {
		views.GDNWq = devicemath.ResidentPtr(base, p.GWq.Off)
		views.GDNWk = devicemath.ResidentPtr(base, p.GWk.Off)
		views.GDNWv = devicemath.ResidentPtr(base, p.GWv.Off)
		views.GDNWbeta = devicemath.ResidentPtr(base, p.GWbeta.Off)
		views.GDNWalpha = devicemath.ResidentPtr(base, p.GWalpha.Off)
		views.GDNWz = devicemath.ResidentPtr(base, p.GWz.Off)
		views.GDNWout = devicemath.ResidentPtr(base, p.GWout.Off)
	} else {
		views.AttnWq = devicemath.ResidentPtr(base, p.Wq.Off)
		views.AttnWk = devicemath.ResidentPtr(base, p.Wk.Off)
		views.AttnWv = devicemath.ResidentPtr(base, p.Wv.Off)
		views.AttnWo = devicemath.ResidentPtr(base, p.Wo.Off)
	}
	return views
}
