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

// TrainDeviceResident runs `steps` Muon updates with the layer weights AND their
// Muon momentum resident on the device across all steps: the layer matrices upload
// once at the start and download only at the final checkpoint -- there is NO per-step
// weight scatter/gather for them, and the Muon Newton-Schulz step runs in place on
// those resident buffers. The whole-stack forward/backward reads the resident
// matrices directly (StackForwardBackwardResidentWeights). The host tail (embedding
// lookup, final RMSNorm, head, softmax-CE) is unchanged, and the sign-updated tensors
// (per-layer norm vectors, final norm) plus the embedding/head Muon tensors run on the
// host optimizer -- exactly the reference host Train trajectory for those groups. The
// only per-step device transfers are activations (embeds in, final out, dOut in) and
// the tiny norm-vector sync/grad; weights and momentum are not among them. Matches
// Train within fp32 tolerance. Attention bias is not supported.
func (m *Model) TrainDeviceResident(worker *device.Worker, tokens []int, steps int, baseLR, mu float64) ([]float64, error) {
	return m.trainDeviceResident(worker, repeatedBatches(tokens, steps), baseLR, mu, false, nil)
}

// TrainDeviceResidentBatches preserves ordered batches under one resident lifecycle.
func (m *Model) TrainDeviceResidentBatches(worker *device.Worker, batches [][]int, baseLR, mu float64) ([]float64, error) {
	return m.trainDeviceResident(worker, batches, baseLR, mu, false, nil)
}

// TrainDeviceResidentFrozenLexical keeps the production frozen lexical tail on
// device; only scalar loss and layer norm gradients cross to host per step.
func (m *Model) TrainDeviceResidentFrozenLexical(worker *device.Worker, tokens []int, steps int, baseLR, mu float64) ([]float64, error) {
	return m.trainDeviceResident(worker, repeatedBatches(tokens, steps), baseLR, mu, true, nil)
}

// TrainDeviceResidentFrozenLexicalBatches preserves ordered minibatches under
// one resident weight, momentum, and allocation lifetime.
func (m *Model) TrainDeviceResidentFrozenLexicalBatches(worker *device.Worker, batches [][]int, baseLR, mu float64) ([]float64, error) {
	return m.trainDeviceResident(worker, batches, baseLR, mu, true, nil)
}

// MeasureDeviceResidentFrozenLexicalBatches returns the resident loop wall;
// admission, upload, checkpoint, and teardown stay outside that phase.
func (m *Model) MeasureDeviceResidentFrozenLexicalBatches(worker *device.Worker, batches [][]int, baseLR, mu float64) ([]float64, time.Duration, error) {
	var wall time.Duration
	trajectory, err := m.trainDeviceResident(worker, batches, baseLR, mu, true, &wall)
	return trajectory, wall, err
}

func repeatedBatches(tokens []int, steps int) [][]int {
	if steps <= 0 {
		return nil
	}
	batches := make([][]int, steps)
	for index := range batches {
		batches[index] = tokens
	}
	return batches
}

func (m *Model) trainDeviceResident(worker *device.Worker, batches [][]int, baseLR, mu float64, frozenLexical bool, loopWall *time.Duration) ([]float64, error) {
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
		return nil, fmt.Errorf("TrainDeviceResident: %s", reason)
	}

	// Derived scale ceiling: how many layers fit fully resident is DERIVED from
	// measured free device memory and the measured allocator granularity (not a
	// supplied n_gpu_layers). Reject up front rather than OOM mid-session.
	capacity, err := m.deriveResidentCapacity(worker, len(tokens), frozenLexical)
	if err != nil {
		return nil, fmt.Errorf("TrainDeviceResident: capacity derivation: %w", err)
	}
	if d.Layers > capacity.Layers {
		return nil, fmt.Errorf("TrainDeviceResident: %d layers exceed derived resident capacity %d (free=%d bytes, G=%d, per-layer=%d bytes)",
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
		return nil, fmt.Errorf("TrainDeviceResident: no layer tensors")
	}
	layerElems := layerEnd - layerStart

	// Per-layer tensor offsets (rebased to the layer block) for the resident stack.
	offsets := make([]devicemath.LayerTensorOffsets, d.Layers)
	for i := 0; i < d.Layers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		offsets[i] = devicemath.LayerTensorOffsets{
			InLN:   offset[p+"input_layernorm.weight"] - layerStart,
			PostLN: offset[p+"post_attention_layernorm.weight"] - layerStart,
			Q:      offset[p+"self_attn.q_proj.weight"] - layerStart,
			K:      offset[p+"self_attn.k_proj.weight"] - layerStart,
			V:      offset[p+"self_attn.v_proj.weight"] - layerStart,
			O:      offset[p+"self_attn.o_proj.weight"] - layerStart,
			Gate:   offset[p+"mlp.gate_proj.weight"] - layerStart,
			Up:     offset[p+"mlp.up_proj.weight"] - layerStart,
			Down:   offset[p+"mlp.down_proj.weight"] - layerStart,
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
	dM, err := devicemath.AllocResidentF32(worker, layerElems, nil)
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
		dFinalM, err = devicemath.AllocResidentF32(worker, d.Hidden, nil)
		if err != nil {
			_ = devicemath.FreeResident(worker, dFinalW, dFinalG)
			return nil, err
		}
		defer func() { _ = devicemath.FreeResident(worker, dFinalW, dFinalG, dFinalM) }()
	}

	invF32 := ropeInvF32(hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim))
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
		normW := make([]devicemath.LayerNormPair, d.Layers)
		for i := 0; i < d.Layers; i++ {
			p := fmt.Sprintf("model.layers.%d.", i)
			io, po := offset[p+"input_layernorm.weight"], offset[p+"post_attention_layernorm.weight"]
			normW[i] = devicemath.LayerNormPair{
				InLN:   weights[io : io+d.Hidden],
				PostLN: weights[po : po+d.Hidden],
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
			gradHead := g.slot(m.headName(), len(head))
			dNormed := make([]float32, seq*d.Hidden)
			hostmath.LinearBackward(dNormed, gradHead, nil, normed, head, dLogits, seq, d.Hidden, d.Vocab, false)
			dx := make([]float32, seq*d.Hidden)
			hostmath.RMSNormBackward(dx, g.slot("model.norm.weight", d.Hidden), final, m.Weights["model.norm.weight"], dNormed, seq, d.Hidden, d.RMSEps, false)
			return dx, nil
		}

		var dxEmbed []float32
		var normGrads []devicemath.LayerNormPair
		if frozenLexical {
			loss, normGrads, err = devicemath.StackForwardBackwardResidentFrozenCausal(
				worker, tokens, dW, dG, offsets, normW, invF32, dHead, dFinalW, dFinalG,
				seq, d.Vocab, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps,
			)
		} else {
			dxEmbed, normGrads, err = devicemath.StackForwardBackwardResidentWeights(
				worker, embeds, dW, dG, offsets, normW, invF32,
				seq, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps, tail,
			)
		}
		if err != nil {
			return nil, err
		}
		trajectory = append(trajectory, loss)

		// Assemble the host gradient buffer for the non-matrix groups only.
		for i := 0; i < d.Layers; i++ {
			p := fmt.Sprintf("model.layers.%d.", i)
			copy(g.slot(p+"input_layernorm.weight", d.Hidden), normGrads[i].InLN)
			copy(g.slot(p+"post_attention_layernorm.weight", d.Hidden), normGrads[i].PostLN)
		}
		if !frozenLexical {
			gradEmbed := g.slot("model.embed_tokens.weight", len(embed))
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
		opt.StepGroups(func(gr optimizer.Group) bool {
			return !isLayerMatrixName(gr.Name) && !(frozenLexical && (gr.Name == m.headName() || gr.Name == "model.norm.weight"))
		})
		updates := make([]optimizer.ResidentMatrix, len(mats))
		for i, mt := range mats {
			updates[i] = optimizer.ResidentMatrix{
				Weights:  devicemath.ResidentPtr(dW, mt.relOff),
				Gradient: devicemath.ResidentPtr(dG, mt.relOff),
				Momentum: devicemath.ResidentPtr(dM, mt.relOff),
				Rows:     mt.rows, Cols: mt.cols,
			}
		}
		if frozenLexical {
			updates = append(updates, optimizer.ResidentMatrix{
				Weights: dFinalW, Gradient: dFinalG, Momentum: dFinalM, Rows: d.Hidden, Cols: 1,
			})
		}
		if err := optimizer.DeviceMuonMatricesResident(worker, updates, step+1, config); err != nil {
			return nil, err
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
	if loopWall != nil {
		*loopWall = time.Since(loopStarted)
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
	scatter(m, names, weights)
	return trajectory, nil
}
