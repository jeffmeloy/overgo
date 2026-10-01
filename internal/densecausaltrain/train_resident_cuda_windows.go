//go:build windows

// Package densecausaltrain trains a dense-causal model resident on the
// device. It holds the device path densecausal's host trainer does not need,
// so the serving packages that link densecausal never reach devicemath; only
// the commands that train import it.
package densecausaltrain

import (
	"fmt"
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
// device across steps. The two per-layer norm vectors (…layernorm.weight) take a
// sign update and stay host-owned, as do the embedding/head/final-norm tensors.
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
// the selected resident policy.
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
	embed, finalNorm, head := m.Weights[embeddingName], m.Weights[finalNormName], m.Weights[headName]

	// Successful allocations admit residency. A free-memory snapshot cannot
	// reserve capacity against concurrent consumers.
	// Host optimizer owns the non-layer-matrix groups; its float64 momentum and the
	// `weights` buffer reproduce the reference host trajectory for those groups.
	opt, err := m.NewDenseUpdate(baseLR, mu, resume)
	if err != nil {
		return nil, err
	}
	weights, _ := opt.Pack.Slabs()
	plan := opt.Pack.Plan()
	initialState := opt.Snapshot()

	// Flat offset of every tensor within the sorted plan buffer.
	offset := make(map[string]int, plan.GroupCount())
	for _, group := range plan.Groups() {
		offset[group.Name] = group.Start
	}

	// Layer block [layerStart, layerEnd): all "model.layers." tensors are contiguous
	// in sorted order. The resident dW/dG/dM cover exactly this block (offsets rebased
	// to layerStart) so no device memory is spent on the host-owned tail tensors.
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
			InLN:     offset[names.InLN] - layerStart,
			PostLN:   offset[names.PostLN] - layerStart,
			Q:        offset[names.Q] - layerStart,
			K:        offset[names.K] - layerStart,
			V:        offset[names.V] - layerStart,
			O:        offset[names.O] - layerStart,
			Gate:     offset[names.Gate] - layerStart,
			Up:       offset[names.Up] - layerStart,
			Down:     offset[names.Down] - layerStart,
			AttnBias: d.AttnBias,
		}
		if d.AttnBias {
			offsets[i].QBias = offset[names.QB] - layerStart
			offsets[i].KBias = offset[names.KB] - layerStart
			offsets[i].VBias = offset[names.VB] - layerStart
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

	// Persistent resident buffers: weights (layer block), gradients, momentum.
	dW, err := devicemath.AllocResidentF32(worker, layerElems, weights[layerStart:layerEnd])
	if err != nil {
		return nil, err
	}
	dG, err := devicemath.AllocResidentF32(worker, layerElems, nil)
	if err != nil {
		_ = devicemath.FreeResident(worker, dW)
		return nil, err
	}
	deviceMomentum := make([]float32, layerElems)
	for index := range deviceMomentum {
		deviceMomentum[index] = float32(initialState.Momentum[layerStart+index])
	}
	dM, err := devicemath.AllocResidentF32(worker, layerElems, deviceMomentum)
	if err != nil {
		_ = devicemath.FreeResident(worker, dW, dG)
		return nil, err
	}
	defer func() { _ = devicemath.FreeResident(worker, dW, dG, dM) }()

	var dHead, dFinalW, dFinalG, dFinalM driver.DevicePtr
	if frozenLexical {
		dHead, err = devicemath.AllocResidentF32(worker, len(head), head)
		if err != nil {
			return nil, err
		}
		defer func() { _ = devicemath.FreeResident(worker, dHead) }()
		dFinalW, err = devicemath.AllocResidentF32(worker, d.Hidden, finalNorm)
		if err != nil {
			return nil, err
		}
		dFinalG, err = devicemath.AllocResidentF32(worker, d.Hidden, nil)
		if err != nil {
			_ = devicemath.FreeResident(worker, dFinalW)
			return nil, err
		}
		finalOffset := offset[finalNormName]
		finalMomentum := make([]float32, d.Hidden)
		for index := range finalMomentum {
			finalMomentum[index] = float32(initialState.Momentum[finalOffset+index])
		}
		dFinalM, err = devicemath.AllocResidentF32(worker, d.Hidden, finalMomentum)
		if err != nil {
			_ = devicemath.FreeResident(worker, dFinalW, dFinalG)
			return nil, err
		}
		defer func() { _ = devicemath.FreeResident(worker, dFinalW, dFinalG, dFinalM) }()
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
	if frozenLexical {
		updates = append(updates, optimizer.ResidentMatrix{
			Weights: dFinalW, Gradient: dFinalG, Momentum: dFinalM, Rows: d.Hidden, Cols: 1,
		})
	}
	residentMuon, err := optimizer.NewResidentMatrixMuonTrainingPlan(worker, updates, initialState.Config)
	if err != nil {
		return nil, err
	}
	defer residentMuon.Close()

	invF32 := dtype.Float64SliceToFloat32(hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim))
	var frozenSession *devicemath.FrozenCausalTrainingSession
	if frozenLexical {
		frozenSession, err = devicemath.NewFrozenCausalTrainingSession(
			worker, dW, dG, offsets, invF32, dHead, dFinalW, dFinalG,
			len(tokens), d.Vocab, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps,
		)
		if err != nil {
			return nil, err
		}
		defer frozenSession.Close()
	}
	vectorWeights := make([]devicemath.LayerTrainVectors, d.Layers)
	for i, names := range layerNames {
		inNorm, postNorm := offset[names.InLN], offset[names.PostLN]
		vectorWeights[i] = devicemath.LayerTrainVectors{
			InLN: weights[inNorm : inNorm+d.Hidden], PostLN: weights[postNorm : postNorm+d.Hidden],
		}
		if d.AttnBias {
			query, key, value := offset[names.QB], offset[names.KB], offset[names.VB]
			vectorWeights[i].QBias = weights[query : query+d.Heads*d.HeadDim]
			vectorWeights[i].KBias = weights[key : key+d.KVHeads*d.HeadDim]
			vectorWeights[i].VBias = weights[value : value+d.KVHeads*d.HeadDim]
		}
	}
	trajectory := make([]float64, 0, steps)

	loopStarted := time.Now()
	for step := range steps {
		stepStarted := time.Now()
		tokens := batches[step]
		// Host oracle tail only; production frozen lexical gathers on device.
		var embeds []float32
		if !frozenLexical {
			embeds = make([]float32, len(tokens)*d.Hidden)
			for token, id := range tokens {
				if id < 0 || id >= d.Vocab {
					return nil, fmt.Errorf("densecausal: token %d out of vocab %d", id, d.Vocab)
				}
				copy(embeds[token*d.Hidden:(token+1)*d.Hidden], embed[id*d.Hidden:(id+1)*d.Hidden])
			}
		}

		// Host head / final RMSNorm / softmax-CE tail, the host trainer's.
		g := densecausal.Grads{}
		var loss float64
		seq := len(tokens)
		tail := func(final []float32) ([]float32, error) {
			normed := make([]float32, seq*d.Hidden)
			hostmath.RMSNormInto(normed, final, finalNorm, seq, d.Hidden, d.RMSEps)
			logits := make([]float32, seq*d.Vocab)
			hostmath.Linear(logits, normed, head, seq, d.Hidden, d.Vocab)
			dLogits := make([]float32, seq*d.Vocab)
			loss = hostmath.SoftmaxCrossEntropy(dLogits[:(seq-1)*d.Vocab], logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)
			gradHead := hostmath.GradientSlot(g, headName, len(head))
			dNormed := make([]float32, seq*d.Hidden)
			hostmath.LinearBackward(dNormed, gradHead, nil, normed, head, dLogits, seq, d.Hidden, d.Vocab, false)
			dx := make([]float32, seq*d.Hidden)
			hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, finalNormName, d.Hidden), final, finalNorm, dNormed, seq, d.Hidden, d.RMSEps, false)
			return dx, nil
		}

		var dxEmbed []float32
		var vectorGrads []devicemath.LayerTrainVectors
		forwardStarted := time.Now()
		if frozenLexical {
			loss, vectorGrads, err = frozenSession.Run(tokens, vectorWeights)
		} else {
			dxEmbed, vectorGrads, err = devicemath.StackForwardBackwardResidentWeights(
				worker, embeds, dW, dG, offsets, vectorWeights, invF32,
				seq, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps, tail,
			)
		}
		if err != nil {
			return nil, err
		}
		if measurement != nil {
			elapsed := time.Since(forwardStarted)
			measurement.ForwardBackward += elapsed
			measurement.ForwardBackwardSteps = append(measurement.ForwardBackwardSteps, elapsed)
		}
		trajectory = append(trajectory, loss)

		// Assemble the host gradient buffer for the non-matrix groups only.
		for i, names := range layerNames {
			copy(hostmath.GradientSlot(g, names.InLN, d.Hidden), vectorGrads[i].InLN)
			copy(hostmath.GradientSlot(g, names.PostLN, d.Hidden), vectorGrads[i].PostLN)
			if d.AttnBias {
				copy(hostmath.GradientSlot(g, names.QB, d.Heads*d.HeadDim), vectorGrads[i].QBias)
				copy(hostmath.GradientSlot(g, names.KB, d.KVHeads*d.HeadDim), vectorGrads[i].KBias)
				copy(hostmath.GradientSlot(g, names.VB, d.KVHeads*d.HeadDim), vectorGrads[i].VBias)
			}
		}
		if !frozenLexical {
			gradEmbed := hostmath.GradientSlot(g, embeddingName, len(embed))
			densecausal.ScatterEmbeddingGradient(gradEmbed, dxEmbed, tokens, d.Hidden)
		}
		for index, group := range plan.Groups() {
			if isLayerMatrixName(group.Name) {
				continue
			}
			if gg, ok := g[group.Name]; ok {
				copy(opt.Pack.Gradients(index), gg)
			} else {
				clear(opt.Pack.Gradients(index))
			}
		}

		// Host Muon: non-layer groups. Device Muon: resident layer matrices.
		hostUpdateStarted := time.Now()
		stepResult := opt.StepGroups(func(gr optimizer.Group) bool {
			return !isLayerMatrixName(gr.Name) && !(frozenLexical && (gr.Name == headName || gr.Name == finalNormName))
		})
		if measurement != nil {
			measurement.HostUpdate += time.Since(hostUpdateStarted)
		}
		deviceUpdateStarted := time.Now()
		if err := residentMuon.StepMatrices(updates, stepResult.Step); err != nil {
			return nil, err
		}
		if measurement != nil {
			elapsed := time.Since(deviceUpdateStarted)
			measurement.DeviceUpdate += elapsed
			measurement.DeviceUpdateSteps = append(measurement.DeviceUpdateSteps, elapsed)
		}

		// Scatter the updated host-owned (non-matrix) weights back into the model for
		// the next step's host embedding lookup and tail. Matrix slots stay device-side.
		for index, group := range plan.Groups() {
			if !isLayerMatrixName(group.Name) && !(frozenLexical && group.Name == finalNormName) {
				copy(m.Weights[group.Name], opt.Pack.Weights(index))
			}
		}
		// The observer sees the FULL step -- forward, backward, host and
		// device updates, scatter -- so a pathological optimizer phase is
		// visible at the first boundary, not after the run.
		if observe != nil {
			if err := observe(step, loss, time.Since(stepStarted)); err != nil {
				return nil, err
			}
		}
	}
	if measurement != nil {
		measurement.Loop = time.Since(loopStarted)
	}

	// Checkpoint: pull the resident matrix weights to host, then scatter everything.
	slices := make([]devicemath.ResidentSlice, len(mats))
	for i, mt := range mats {
		slices[i] = devicemath.ResidentSlice{ElemOffset: mt.relOff, Data: weights[mt.absOff : mt.absOff+mt.size]}
	}
	if err := devicemath.ReadResident(worker, dW, slices...); err != nil {
		return nil, err
	}
	if frozenLexical {
		finalOffset := offset[finalNormName]
		if err := devicemath.ReadResident(worker, dFinalW, devicemath.ResidentSlice{Data: weights[finalOffset : finalOffset+d.Hidden]}); err != nil {
			return nil, err
		}
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
	if frozenLexical {
		finalOffset := offset[finalNormName]
		finalMomentum := make([]float32, d.Hidden)
		if err := devicemath.ReadResident(worker, dFinalM, devicemath.ResidentSlice{Data: finalMomentum}); err != nil {
			return nil, err
		}
		for index, value := range finalMomentum {
			state.Momentum[finalOffset+index] = float64(value)
		}
	}
	if stateOut != nil {
		*stateOut = state
	}
	opt.Pack.Scatter()
	return trajectory, nil
}
