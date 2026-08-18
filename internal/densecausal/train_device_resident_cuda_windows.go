//go:build windows

package densecausal

import (
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

// perLayerWeightElems is one layer's total weight element count (nine matrices +
// two norm vectors), read from layer 0's tensors -- the residency unit's weight
// footprint for the capacity derivation.
func (m *Model) perLayerWeightElems() int {
	total := 0
	for name, w := range m.Weights {
		if strings.HasPrefix(name, "model.layers.0.") {
			total += len(w)
		}
	}
	return total
}

// DeriveResidentCapacity measures free device memory and the allocator
// granularity, then derives how many transformer layers of this model can be
// trained fully resident on the device at sequence length seq -- the scale
// ceiling computed from measured bytes, not a supplied n_gpu_layers (closes
// scale-ceiling-measure / YOINK-1).
func (m *Model) DeriveResidentCapacity(worker *device.Worker, seq int) (devicemath.ResidentCapacity, error) {
	return m.deriveResidentCapacity(worker, seq, false)
}

func (m *Model) deriveResidentCapacity(worker *device.Worker, seq int, frozenLexical bool) (devicemath.ResidentCapacity, error) {
	d := m.Dims
	g, err := devicemath.MeasureAllocGranularity(worker)
	if err != nil {
		return devicemath.ResidentCapacity{}, err
	}
	free, err := devicemath.MeasureFreeBytes(worker)
	if err != nil {
		return devicemath.ResidentCapacity{}, err
	}
	fixed, perLayer := devicemath.ResidentLayerPlanSizes(
		seq, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, m.perLayerWeightElems())
	if frozenLexical {
		fixed = append(fixed, devicemath.FrozenCausalTailPlanSizes(seq, d.Vocab, d.Hidden)...)
	}
	return devicemath.DeriveResidentCapacity(free, g, fixed, perLayer), nil
}

type DeviceTrainingOptions struct {
	FrozenLexical bool
	Measure       bool
	Resume        *TrainState
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
	result.Losses, err = m.trainDeviceResident(worker, batches, baseLR, mu, options.FrozenLexical, measurement, options.Resume, &result.State)
	return result, err
}

func (m *Model) trainDeviceResident(worker *device.Worker, batches [][]int, baseLR, mu float64, frozenLexical bool, measurement *DeviceTrainingMeasurement, resume *TrainState, stateOut *TrainState) ([]float64, error) {
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

	// Derived scale ceiling: how many layers fit fully resident is DERIVED from
	// measured free device memory and the measured allocator granularity (not a
	// supplied n_gpu_layers). Reject up front rather than OOM mid-session.
	capacity, err := m.deriveResidentCapacity(worker, len(tokens), frozenLexical)
	if err != nil {
		return nil, fmt.Errorf("densecausal resident training: capacity derivation: %w", err)
	}
	if d.Layers > capacity.Layers {
		return nil, fmt.Errorf("densecausal resident training: %d layers exceed derived resident capacity %d (free=%d bytes, G=%d, per-layer=%d bytes)",
			d.Layers, capacity.Layers, capacity.FreeBytes, capacity.Granularity, capacity.PerLayerBytes)
	}

	names, weights, gradients, plan, resolvedLR, err := m.trainSetup(baseLR)
	if err != nil {
		return nil, err
	}
	config := optimizer.Config{BaseLearningRate: resolvedLR, Momentum: mu, Schedule: optimizer.ScheduleConstant}
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
			if end > layerEnd {
				layerEnd = end
			}
		}
	}
	if layerStart < 0 {
		return nil, fmt.Errorf("densecausal resident training: no layer tensors")
	}
	layerElems := layerEnd - layerStart

	// Per-layer tensor offsets (rebased to the layer block) for the resident stack.
	offsets := make([]devicemath.LayerTensorOffsets, d.Layers)
	for i := 0; i < d.Layers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		offsets[i] = devicemath.LayerTensorOffsets{
			InLN:     offset[p+"input_layernorm.weight"] - layerStart,
			PostLN:   offset[p+"post_attention_layernorm.weight"] - layerStart,
			Q:        offset[p+"self_attn.q_proj.weight"] - layerStart,
			K:        offset[p+"self_attn.k_proj.weight"] - layerStart,
			V:        offset[p+"self_attn.v_proj.weight"] - layerStart,
			O:        offset[p+"self_attn.o_proj.weight"] - layerStart,
			Gate:     offset[p+"mlp.gate_proj.weight"] - layerStart,
			Up:       offset[p+"mlp.up_proj.weight"] - layerStart,
			Down:     offset[p+"mlp.down_proj.weight"] - layerStart,
			AttnBias: d.AttnBias,
		}
		if d.AttnBias {
			offsets[i].QBias = offset[p+"self_attn.q_proj.bias"] - layerStart
			offsets[i].KBias = offset[p+"self_attn.k_proj.bias"] - layerStart
			offsets[i].VBias = offset[p+"self_attn.v_proj.bias"] - layerStart
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
		dHead, err = devicemath.AllocResidentF32(worker, len(m.head()), m.head())
		if err != nil {
			return nil, err
		}
		defer func() { _ = devicemath.FreeResident(worker, dHead) }()
		dFinalW, err = devicemath.AllocResidentF32(worker, d.Hidden, m.Weights["model.norm.weight"])
		if err != nil {
			return nil, err
		}
		dFinalG, err = devicemath.AllocResidentF32(worker, d.Hidden, nil)
		if err != nil {
			_ = devicemath.FreeResident(worker, dFinalW)
			return nil, err
		}
		finalOffset := offset["model.norm.weight"]
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
	trajectory := make([]float64, 0, steps)

	loopStarted := time.Now()
	for step := 0; step < steps; step++ {
		tokens := batches[step]
		// Host oracle tail only; production frozen lexical gathers on device.
		embed := m.Weights["model.embed_tokens.weight"]
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

		// Current host-owned norm vectors (from the flat weights buffer) to sync into
		// the resident weight buffer before the forward reads them.
		vectorWeights := make([]devicemath.LayerTrainVectors, d.Layers)
		for i := 0; i < d.Layers; i++ {
			p := fmt.Sprintf("model.layers.%d.", i)
			io, po := offset[p+"input_layernorm.weight"], offset[p+"post_attention_layernorm.weight"]
			vectorWeights[i] = devicemath.LayerTrainVectors{
				InLN:   weights[io : io+d.Hidden],
				PostLN: weights[po : po+d.Hidden],
			}
			if d.AttnBias {
				qo, ko, vo := offset[p+"self_attn.q_proj.bias"], offset[p+"self_attn.k_proj.bias"], offset[p+"self_attn.v_proj.bias"]
				vectorWeights[i].QBias = weights[qo : qo+d.Heads*d.HeadDim]
				vectorWeights[i].KBias = weights[ko : ko+d.KVHeads*d.HeadDim]
				vectorWeights[i].VBias = weights[vo : vo+d.KVHeads*d.HeadDim]
			}
		}

		// Host head / final RMSNorm / softmax-CE tail (unchanged from deviceLossAndGrads).
		g := Grads{}
		var loss float64
		seq := len(tokens)
		tail := func(final []float32) ([]float32, error) {
			normed := make([]float32, seq*d.Hidden)
			hostmath.RMSNormInto(normed, final, m.Weights["model.norm.weight"], seq, d.Hidden, d.RMSEps)
			head := m.head()
			logits := make([]float32, seq*d.Vocab)
			hostmath.Linear(logits, normed, head, seq, d.Hidden, d.Vocab)
			dLogits := make([]float32, seq*d.Vocab)
			loss = hostmath.SoftmaxCrossEntropy(dLogits[:(seq-1)*d.Vocab], logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)
			gradHead := hostmath.GradientSlot(g, m.headName(), len(head))
			dNormed := make([]float32, seq*d.Hidden)
			hostmath.LinearBackward(dNormed, gradHead, nil, normed, head, dLogits, seq, d.Hidden, d.Vocab, false)
			dx := make([]float32, seq*d.Hidden)
			hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, "model.norm.weight", d.Hidden), final, m.Weights["model.norm.weight"], dNormed, seq, d.Hidden, d.RMSEps, false)
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
		for i := 0; i < d.Layers; i++ {
			p := fmt.Sprintf("model.layers.%d.", i)
			copy(hostmath.GradientSlot(g, p+"input_layernorm.weight", d.Hidden), vectorGrads[i].InLN)
			copy(hostmath.GradientSlot(g, p+"post_attention_layernorm.weight", d.Hidden), vectorGrads[i].PostLN)
			if d.AttnBias {
				copy(hostmath.GradientSlot(g, p+"self_attn.q_proj.bias", d.Heads*d.HeadDim), vectorGrads[i].QBias)
				copy(hostmath.GradientSlot(g, p+"self_attn.k_proj.bias", d.KVHeads*d.HeadDim), vectorGrads[i].KBias)
				copy(hostmath.GradientSlot(g, p+"self_attn.v_proj.bias", d.KVHeads*d.HeadDim), vectorGrads[i].VBias)
			}
		}
		if !frozenLexical {
			gradEmbed := hostmath.GradientSlot(g, "model.embed_tokens.weight", len(embed))
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
			return !isLayerMatrixName(gr.Name) && !(frozenLexical && (gr.Name == m.headName() || gr.Name == "model.norm.weight"))
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
			if !isLayerMatrixName(name) && !(frozenLexical && name == "model.norm.weight") {
				copy(m.Weights[name], weights[off:off+n])
			}
			off += n
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
		finalOffset := offset["model.norm.weight"]
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
		finalOffset := offset["model.norm.weight"]
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
