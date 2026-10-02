//go:build windows

// Package densecausaltrain trains a dense-causal model resident on the
// device. It holds the device path densecausal's host trainer does not need,
// so the serving packages that link densecausal never reach devicemath; only
// the commands that train import it.
package densecausaltrain

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/densecausal"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/tensor/dtype"
)

// isLayerMatrixName reports whether a tensor is a per-layer projection matrix
// (q/k/v/o/gate/up/down) -- the Muon-updated layer weights that live resident on the
// device across steps. The per-layer norm vectors (…layernorm.weight) and
// attention biases take their Muon step on the host; the embedding, head and
// final norm live on the device as well.
func isLayerMatrixName(name string) bool {
	return strings.HasPrefix(name, "model.layers.") && strings.HasSuffix(name, "_proj.weight")
}

// Options selects the resident policy: a frozen lexical table, phase
// measurement, the state a resume continues from, and a step observer.
type Options struct {
	FrozenLexical bool
	Measure       bool
	Resume        *densecausal.TrainState
	// Observe receives each completed step with its measured wall; a non-nil
	// return aborts at the step boundary.
	Observe densecausal.TrainObserver
}

// Measurement separates synchronized resident phases.
type Measurement struct {
	ForwardBackward      time.Duration
	HostUpdate           time.Duration
	DeviceUpdate         time.Duration
	Loop                 time.Duration
	ForwardBackwardSteps []time.Duration
	DeviceUpdateSteps    []time.Duration
}

// Result is a resident run's loss trajectory, the optimizer state its last
// step left, and its phase measurement when one was asked for.
type Result struct {
	Losses      []float64
	State       densecausal.TrainState
	Measurement Measurement
}

// TrainDeviceResident trains m on worker, one Muon step per batch, under
// the selected resident policy. The forward, the backward and the lexical
// tail run on the device; only the layer norm and bias vectors take their
// Muon step on the host.
func TrainDeviceResident(m *densecausal.Model, worker *device.Worker, batches [][]int, baseLR, mu float64, options Options) (Result, error) {
	result := Result{}
	var measurement *Measurement
	if options.Measure {
		measurement = &result.Measurement
	}
	var err error
	result.Losses, err = trainDeviceResident(m, worker, batches, baseLR, mu, options.FrozenLexical, measurement, options.Resume, &result.State, options.Observe)
	return result, err
}

func trainDeviceResident(m *densecausal.Model, worker *device.Worker, batches [][]int, baseLR, mu float64, frozenLexical bool, measurement *Measurement, resume *densecausal.TrainState, stateOut *densecausal.TrainState, observe densecausal.TrainObserver) ([]float64, error) {
	if len(batches) == 0 || len(batches[0]) <= 1 {
		return nil, fmt.Errorf("densecausal: need at least one batch with two tokens")
	}
	tokens := batches[0]
	for index, batch := range batches {
		if len(batch) != len(tokens) {
			return nil, fmt.Errorf("densecausal: batch %d length %d != %d", index, len(batch), len(tokens))
		}
	}
	steps := len(batches)
	d := m.Dims
	if ok, reason := densecausal.DeviceTrainingSupported(d); !ok {
		return nil, fmt.Errorf("densecausal resident training: %s", reason)
	}
	embeddingName, finalNormName, headName := m.LexicalTensorNames()
	tied := embeddingName == headName

	// Successful allocations admit residency. A free-memory snapshot cannot
	// reserve capacity against concurrent consumers.
	// Host optimizer owns the layer vector groups; its float64 momentum and the
	// `weights` buffer reproduce the reference host trajectory for those groups.
	opt, err := m.NewDenseUpdate(baseLR, mu, resume)
	if err != nil {
		return nil, err
	}
	weights, _ := opt.Pack.Slabs()
	plan := opt.Pack.Plan()
	initialState := opt.Snapshot()
	momentumF32 := func(start, end int) []float32 {
		return dtype.Float64SliceToFloat32(initialState.Momentum[start:end])
	}

	// Every group's index within the sorted plan buffer, by name.
	groupIndex := make(map[string]int, plan.GroupCount())
	for index, group := range plan.Groups() {
		groupIndex[group.Name] = index
	}
	group := func(name string) optimizer.Group { return plan.Groups()[groupIndex[name]] }
	offset := func(name string) int { return group(name).Start }

	// Layer block [layerStart, layerEnd): all "model.layers." tensors are contiguous
	// in sorted order. The resident dW/dG/dM cover exactly this block (offsets rebased
	// to layerStart); the lexical tensors take their own resident buffers.
	layerStart, layerEnd := -1, 0
	for _, group := range plan.Groups() {
		if strings.HasPrefix(group.Name, "model.layers.") {
			if layerStart < 0 {
				layerStart = group.Start
			}
			layerEnd = max(layerEnd, group.End)
		}
	}
	if layerStart < 0 {
		return nil, fmt.Errorf("densecausal resident training: no layer tensors")
	}
	layerElems := layerEnd - layerStart

	// Per-layer tensor offsets (rebased to the layer block) for the resident stack.
	offsets := make([]devicemath.LayerTensorOffsets, d.Layers)
	layerNames := make([]densecausal.LayerTensorNames, d.Layers)
	for i := range layerNames {
		names := densecausal.DenseLayerTensorNames(i)
		layerNames[i] = names
		offsets[i] = devicemath.LayerTensorOffsets{
			InLN:     offset(names.InLN) - layerStart,
			PostLN:   offset(names.PostLN) - layerStart,
			Q:        offset(names.Q) - layerStart,
			K:        offset(names.K) - layerStart,
			V:        offset(names.V) - layerStart,
			O:        offset(names.O) - layerStart,
			Gate:     offset(names.Gate) - layerStart,
			Up:       offset(names.Up) - layerStart,
			Down:     offset(names.Down) - layerStart,
			AttnBias: d.AttnBias,
		}
		if d.AttnBias {
			offsets[i].QBias = offset(names.QB) - layerStart
			offsets[i].KBias = offset(names.KB) - layerStart
			offsets[i].VBias = offset(names.VB) - layerStart
		}
	}

	// Matrix Muon groups from the plan (device pointers filled once buffers exist).
	type matGroup struct {
		relOff, absOff, size, rows, cols int
	}
	var mats []matGroup
	for gi := 0; gi < plan.GroupCount(); gi++ {
		g, _ := plan.Group(gi)
		if isLayerMatrixName(g.Name) {
			mats = append(mats, matGroup{
				relOff: g.Start - layerStart, absOff: g.Start, size: g.End - g.Start, rows: g.Rows, cols: g.Cols,
			})
		}
	}

	// Persistent resident buffers, released together once every session
	// that reads them has closed.
	var allocated []driver.DevicePtr
	defer func() { _ = devicemath.FreeResident(worker, allocated...) }()
	resident := func(count int, initial []float32) (driver.DevicePtr, error) {
		pointer, err := devicemath.AllocResidentF32(worker, count, initial)
		if err == nil {
			allocated = append(allocated, pointer)
		}
		return pointer, err
	}

	// Layer block: weights, gradients, momentum.
	dW, err := resident(layerElems, weights[layerStart:layerEnd])
	if err != nil {
		return nil, err
	}
	dG, err := resident(layerElems, nil)
	if err != nil {
		return nil, err
	}
	deviceMomentum := momentumF32(layerStart, layerEnd)
	dM, err := resident(layerElems, deviceMomentum)
	if err != nil {
		return nil, err
	}

	// Lexical tensors: the weights the step reads and, when they train, the
	// gradient and momentum their device Muon step keeps.
	lexicalMatrix := func(name string, trained bool) (optimizer.ResidentMatrix, error) {
		spec := group(name)
		size := spec.End - spec.Start
		matrix := optimizer.ResidentMatrix{Rows: spec.Rows, Cols: spec.Cols}
		var err error
		if matrix.Weights, err = resident(size, weights[spec.Start:spec.End]); err != nil || !trained {
			return matrix, err
		}
		if matrix.Gradient, err = resident(size, nil); err != nil {
			return matrix, err
		}
		matrix.Momentum, err = resident(size, momentumF32(spec.Start, spec.End))
		return matrix, err
	}
	embedding, err := lexicalMatrix(embeddingName, !frozenLexical)
	if err != nil {
		return nil, err
	}
	head := embedding
	if !tied {
		if head, err = lexicalMatrix(headName, !frozenLexical); err != nil {
			return nil, err
		}
	}
	finalNorm, err := lexicalMatrix(finalNormName, true)
	if err != nil {
		return nil, err
	}

	updates := make([]optimizer.ResidentMatrix, len(mats))
	for i, matrix := range mats {
		updates[i] = optimizer.ResidentMatrix{
			Weights:  devicemath.ResidentPtr(dW, matrix.relOff),
			Gradient: devicemath.ResidentPtr(dG, matrix.relOff),
			Momentum: devicemath.ResidentPtr(dM, matrix.relOff),
			Rows:     matrix.rows, Cols: matrix.cols,
		}
	}
	// The device steps the final norm and, when the table trains, the
	// embedding and head. The frozen protocol steps the final norm under the
	// layers' matched training schedule; a trained table keeps the host
	// step's Newton-Schulz schedule for its lexical tensors, which the device
	// step reproduces.
	lexical := map[string]optimizer.ResidentMatrix{embeddingName: embedding, headName: head, finalNormName: finalNorm}
	deviceStepped := []string{finalNormName}
	var lexicalUpdates []optimizer.ResidentMatrix
	if frozenLexical {
		updates = append(updates, finalNorm)
	} else {
		deviceStepped = slices.Sorted(maps.Keys(lexical))
		for _, name := range deviceStepped {
			lexicalUpdates = append(lexicalUpdates, lexical[name])
		}
	}
	residentMuon, err := optimizer.NewResidentMatrixMuonTrainingPlan(worker, updates, initialState.Config)
	if err != nil {
		return nil, err
	}
	defer residentMuon.Close()
	var lexicalMuon *optimizer.ResidentMuonPlan
	if len(lexicalUpdates) > 0 {
		if lexicalMuon, err = optimizer.NewResidentMatrixMuonPlan(worker, lexicalUpdates, initialState.Config); err != nil {
			return nil, err
		}
		defer lexicalMuon.Close()
	}

	invF32 := dtype.Float64SliceToFloat32(hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim))
	session, err := devicemath.NewCausalTrainingSession(
		worker, dW, dG, offsets, invF32,
		devicemath.CausalLexical{
			Embedding: embedding.Weights, Head: head.Weights, FinalNorm: finalNorm.Weights,
			EmbeddingGrad: embedding.Gradient, HeadGrad: head.Gradient, FinalNormGrad: finalNorm.Gradient,
		},
		len(tokens), d.Vocab, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps,
	)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	vectorWeights := make([]devicemath.LayerTrainVectors, d.Layers)
	for i, names := range layerNames {
		inNorm, postNorm := offset(names.InLN), offset(names.PostLN)
		vectorWeights[i] = devicemath.LayerTrainVectors{
			InLN: weights[inNorm : inNorm+d.Hidden], PostLN: weights[postNorm : postNorm+d.Hidden],
		}
		if d.AttnBias {
			query, key, value := offset(names.QB), offset(names.KB), offset(names.VB)
			vectorWeights[i].QBias = weights[query : query+d.Heads*d.HeadDim]
			vectorWeights[i].KBias = weights[key : key+d.KVHeads*d.HeadDim]
			vectorWeights[i].VBias = weights[value : value+d.KVHeads*d.HeadDim]
		}
	}
	hostOwned := func(name string) bool {
		_, deviceOwned := lexical[name]
		return !isLayerMatrixName(name) && !deviceOwned
	}
	hostGradient := func(name string) []float32 { return opt.Pack.Gradients(groupIndex[name]) }
	trajectory := make([]float64, 0, steps)

	loopStarted := time.Now()
	for step := range steps {
		stepStarted := time.Now()
		forwardStarted := time.Now()
		loss, vectorGrads, err := session.Run(batches[step], vectorWeights)
		if err != nil {
			return nil, err
		}
		if measurement != nil {
			elapsed := time.Since(forwardStarted)
			measurement.ForwardBackward += elapsed
			measurement.ForwardBackwardSteps = append(measurement.ForwardBackwardSteps, elapsed)
		}
		trajectory = append(trajectory, loss)

		// The layer vector gradients are the host optimizer's only inputs.
		for i, names := range layerNames {
			copy(hostGradient(names.InLN), vectorGrads[i].InLN)
			copy(hostGradient(names.PostLN), vectorGrads[i].PostLN)
			if d.AttnBias {
				copy(hostGradient(names.QB), vectorGrads[i].QBias)
				copy(hostGradient(names.KB), vectorGrads[i].KBias)
				copy(hostGradient(names.VB), vectorGrads[i].VBias)
			}
		}

		// Host Muon: layer vectors, which the session reads from the packed
		// weights. Device Muon: layer matrices and the lexical tensors.
		hostUpdateStarted := time.Now()
		stepResult := opt.StepGroups(func(group optimizer.Group) bool { return hostOwned(group.Name) })
		if measurement != nil {
			measurement.HostUpdate += time.Since(hostUpdateStarted)
		}
		deviceUpdateStarted := time.Now()
		if err := residentMuon.StepMatrices(updates, stepResult.Step); err != nil {
			return nil, err
		}
		if lexicalMuon != nil {
			if err := lexicalMuon.StepMatrices(lexicalUpdates, stepResult.Step); err != nil {
				return nil, err
			}
		}
		if measurement != nil {
			elapsed := time.Since(deviceUpdateStarted)
			measurement.DeviceUpdate += elapsed
			measurement.DeviceUpdateSteps = append(measurement.DeviceUpdateSteps, elapsed)
		}
		// The observer sees the FULL step -- forward, backward, host and
		// device updates -- so a pathological optimizer phase is visible at
		// the first boundary, not after the run.
		if observe != nil {
			if err := observe(step, loss, time.Since(stepStarted)); err != nil {
				return nil, err
			}
		}
	}
	if measurement != nil {
		measurement.Loop = time.Since(loopStarted)
	}

	// Checkpoint: pull the device-owned weights and momentum into the pack,
	// then scatter everything.
	weightSlices := make([]devicemath.ResidentSlice, len(mats))
	for i, mt := range mats {
		weightSlices[i] = devicemath.ResidentSlice{ElemOffset: mt.relOff, Data: weights[mt.absOff : mt.absOff+mt.size]}
	}
	if err := devicemath.ReadResident(worker, dW, weightSlices...); err != nil {
		return nil, err
	}
	state := opt.Snapshot()
	momentumSlices := make([]devicemath.ResidentSlice, len(mats))
	for index, matrix := range mats {
		momentumSlices[index] = devicemath.ResidentSlice{
			ElemOffset: matrix.relOff,
			Data:       deviceMomentum[matrix.relOff : matrix.relOff+matrix.size],
		}
	}
	if err := devicemath.ReadResident(worker, dM, momentumSlices...); err != nil {
		return nil, err
	}
	for _, matrix := range mats {
		for index := range matrix.size {
			state.Momentum[matrix.absOff+index] = float64(deviceMomentum[matrix.relOff+index])
		}
	}
	for _, name := range deviceStepped {
		spec := group(name)
		if err := devicemath.ReadResident(worker, lexical[name].Weights, devicemath.ResidentSlice{Data: weights[spec.Start:spec.End]}); err != nil {
			return nil, err
		}
		momentum := make([]float32, spec.End-spec.Start)
		if err := devicemath.ReadResident(worker, lexical[name].Momentum, devicemath.ResidentSlice{Data: momentum}); err != nil {
			return nil, err
		}
		for position, value := range momentum {
			state.Momentum[spec.Start+position] = float64(value)
		}
	}
	if stateOut != nil {
		*stateOut = state
	}
	opt.Pack.Scatter()
	return trajectory, nil
}
