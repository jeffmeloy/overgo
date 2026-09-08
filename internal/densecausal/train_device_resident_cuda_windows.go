//go:build windows

package densecausal

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
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

type DeviceTrainingOptions struct {
	FrozenLexical bool
	Measure       bool
	Resume        *TrainState
	// Observe receives each completed step with its measured wall; a non-nil
	// return aborts at the step boundary.
	Observe TrainObserver
}

// DeviceTrainingMeasurement separates synchronized resident phases.
type DeviceTrainingMeasurement struct {
	ForwardBackward      time.Duration
	HostUpdate           time.Duration
	DeviceUpdate         time.Duration
	Loop                 time.Duration
	ForwardBackwardSteps []time.Duration
	DeviceUpdateSteps    []time.Duration
}

type DeviceTrainingResult struct {
	Losses      []float64
	State       TrainState
	Measurement DeviceTrainingMeasurement
}

// TrainDeviceResident runs the selected resident policy.
func (m *Model) TrainDeviceResident(worker *device.Worker, batches [][]int, baseLR, mu float64, options DeviceTrainingOptions) (DeviceTrainingResult, error) {
	result := DeviceTrainingResult{}
	var measurement *DeviceTrainingMeasurement
	if options.Measure {
		measurement = &result.Measurement
	}
	var err error
	result.Losses, err = m.trainDeviceResident(worker, batches, baseLR, mu, options.FrozenLexical, measurement, options.Resume, &result.State, options.Observe)
	return result, err
}

func (m *Model) trainDeviceResident(worker *device.Worker, batches [][]int, baseLR, mu float64, frozenLexical bool, measurement *DeviceTrainingMeasurement, resume *TrainState, stateOut *TrainState, observe TrainObserver) ([]float64, error) {
	if len(batches) == 0 || len(batches[0]) < 2 {
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
	if ok, reason := DeviceTrainingSupported(d); !ok {
		return nil, fmt.Errorf("densecausal resident training: %s", reason)
	}

	// Successful allocations admit residency. A free-memory snapshot cannot
	// reserve capacity against concurrent consumers.
	names, weights, gradients, plan, err := m.trainSetup()
	if err != nil {
		return nil, err
	}
	if baseLR <= 0 {
		return nil, errors.New("densecausal: positive recipe-derived learning rate required")
	}
	config := optimizer.Config{BaseLearningRate: baseLR, Momentum: mu, Schedule: optimizer.ScheduleConstant}
	// Host optimizer owns the non-layer-matrix groups; its float64 momentum and the
	// `weights` buffer reproduce the reference host trajectory for those groups.
	opt, err := optimizer.New(weights, gradients, plan, config)
	if err != nil {
		return nil, err
	}
	if resume != nil {
		if err := opt.Restore(*resume); err != nil {
			return nil, err
		}
	}
	initialState := opt.Snapshot()

	// Flat offset of every tensor within the sorted plan buffer.
	offset := make(map[string]int, len(names))
	acc := 0
	for _, name := range names {
		offset[name] = acc
		acc += len(m.Weights[name])
	}

	// Layer block [layerStart, layerEnd): all "model.layers." tensors are contiguous
	// in sorted order. The resident dW/dG/dM cover exactly this block (offsets rebased
	// to layerStart) so no device memory is spent on the host-owned tail tensors.
	layerStart, layerEnd := -1, 0
	for _, name := range names {
		if strings.HasPrefix(name, "model.layers.") {
			if layerStart < 0 {
				layerStart = offset[name]
			}
			end := offset[name] + len(m.Weights[name])
			layerEnd = max(layerEnd, end)
		}
	}
	if layerStart < 0 {
		return nil, fmt.Errorf("densecausal resident training: no layer tensors")
	}
	layerElems := layerEnd - layerStart

	// Per-layer tensor offsets (rebased to the layer block) for the resident stack.
	offsets := make([]devicemath.LayerTensorOffsets, d.Layers)
	for i, layer := range m.layers {
		names := layer.names
		offsets[i] = devicemath.LayerTensorOffsets{
			InLN:     offset[names.inLN] - layerStart,
			PostLN:   offset[names.postLN] - layerStart,
			Q:        offset[names.q] - layerStart,
			K:        offset[names.k] - layerStart,
			V:        offset[names.v] - layerStart,
			O:        offset[names.o] - layerStart,
			Gate:     offset[names.gate] - layerStart,
			Up:       offset[names.up] - layerStart,
			Down:     offset[names.down] - layerStart,
			AttnBias: d.AttnBias,
		}
		if d.AttnBias {
			offsets[i].QBias = offset[names.qb] - layerStart
			offsets[i].KBias = offset[names.kb] - layerStart
			offsets[i].VBias = offset[names.vb] - layerStart
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
		dHead, err = devicemath.AllocResidentF32(worker, len(m.tensors.head.values), m.tensors.head.values)
		if err != nil {
			return nil, err
		}
		defer func() { _ = devicemath.FreeResident(worker, dHead) }()
		dFinalW, err = devicemath.AllocResidentF32(worker, d.Hidden, m.tensors.finalNorm.values)
		if err != nil {
			return nil, err
		}
		dFinalG, err = devicemath.AllocResidentF32(worker, d.Hidden, nil)
		if err != nil {
			_ = devicemath.FreeResident(worker, dFinalW)
			return nil, err
		}
		finalOffset := offset[m.tensors.finalNorm.name]
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
	residentMuon, err := optimizer.NewResidentMatrixMuonTrainingPlan(worker, updates, config)
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
	for i, layer := range m.layers {
		names := layer.names
		inNorm, postNorm := offset[names.inLN], offset[names.postLN]
		vectorWeights[i] = devicemath.LayerTrainVectors{
			InLN: weights[inNorm : inNorm+d.Hidden], PostLN: weights[postNorm : postNorm+d.Hidden],
		}
		if d.AttnBias {
			query, key, value := offset[names.qb], offset[names.kb], offset[names.vb]
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
		embed := m.tensors.embedding.values
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

		// Host head / final RMSNorm / softmax-CE tail (unchanged from deviceLossAndGrads).
		g := Grads{}
		var loss float64
		seq := len(tokens)
		tail := func(final []float32) ([]float32, error) {
			normed := make([]float32, seq*d.Hidden)
			hostmath.RMSNormInto(normed, final, m.tensors.finalNorm.values, seq, d.Hidden, d.RMSEps)
			head := m.tensors.head.values
			logits := make([]float32, seq*d.Vocab)
			hostmath.Linear(logits, normed, head, seq, d.Hidden, d.Vocab)
			dLogits := make([]float32, seq*d.Vocab)
			loss = hostmath.SoftmaxCrossEntropy(dLogits[:(seq-1)*d.Vocab], logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)
			gradHead := hostmath.GradientSlot(g, m.tensors.head.name, len(head))
			dNormed := make([]float32, seq*d.Hidden)
			hostmath.LinearBackward(dNormed, gradHead, nil, normed, head, dLogits, seq, d.Hidden, d.Vocab, false)
			dx := make([]float32, seq*d.Hidden)
			hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, m.tensors.finalNorm.name, d.Hidden), final, m.tensors.finalNorm.values, dNormed, seq, d.Hidden, d.RMSEps, false)
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
		for i, layer := range m.layers {
			names := layer.names
			copy(hostmath.GradientSlot(g, names.inLN, d.Hidden), vectorGrads[i].InLN)
			copy(hostmath.GradientSlot(g, names.postLN, d.Hidden), vectorGrads[i].PostLN)
			if d.AttnBias {
				copy(hostmath.GradientSlot(g, names.qb, d.Heads*d.HeadDim), vectorGrads[i].QBias)
				copy(hostmath.GradientSlot(g, names.kb, d.KVHeads*d.HeadDim), vectorGrads[i].KBias)
				copy(hostmath.GradientSlot(g, names.vb, d.KVHeads*d.HeadDim), vectorGrads[i].VBias)
			}
		}
		if !frozenLexical {
			gradEmbed := hostmath.GradientSlot(g, m.tensors.embedding.name, len(embed))
			scatterEmbeddingGradient(gradEmbed, dxEmbed, tokens, d.Hidden)
		}
		off := 0
		for _, name := range names {
			n := len(m.Weights[name])
			if !isLayerMatrixName(name) {
				if gg, ok := g[name]; ok {
					copy(gradients[off:off+n], gg)
				} else {
					clear(gradients[off : off+n])
				}
			}
			off += n
		}

		// Host Muon: non-layer groups. Device Muon: resident layer matrices.
		hostUpdateStarted := time.Now()
		stepResult := opt.StepGroups(func(gr optimizer.Group) bool {
			return !isLayerMatrixName(gr.Name) && !(frozenLexical && (gr.Name == m.tensors.head.name || gr.Name == m.tensors.finalNorm.name))
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
		off = 0
		for _, name := range names {
			n := len(m.Weights[name])
			if !isLayerMatrixName(name) && !(frozenLexical && name == m.tensors.finalNorm.name) {
				copy(m.Weights[name], weights[off:off+n])
			}
			off += n
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
		finalOffset := offset[m.tensors.finalNorm.name]
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
		finalOffset := offset[m.tensors.finalNorm.name]
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
	scatter(m, names, weights)
	return trajectory, nil
}
