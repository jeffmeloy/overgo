// Full modality-transformer trainer for branch-routed prompt stacks: the
// vision-branch fork of every layer (attention projections, SiLU MLP, and the
// two branch norms) trains as f32 masters decoded from the BF16 serving
// storage, over the shared Muon stepper with fully derived hyperparameters.
// The forward is the training-precision (f32, unrounded) mirror of
// PromptLayerForward — branch routing, segment-window attention, rope and
// per-head QK-norm included — and the objective is the pipeline's own
// contract: next-token cross-entropy through the final norm and the tied
// head at the prompt's text positions. The frozen text branch, QK-norm
// scales, final norm, and head participate in the graph (gradient flows
// through them) without weight updates. This closes the organ cells' named
// full-pipeline promotion gate for the modality transformer.
package routedlm

import (
	"fmt"
	"math"
	"slices"
	"sync"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// MoTTarget: one supervised position — the prompt row whose next token is
// predicted and that next token's id.
type MoTTarget struct {
	Position int
	Token    int
}

// MoTTrainStepResult: measured facts of one observed full-pipeline step.
type MoTTrainStepResult = optimizer.ObservedStepResult

// motSpans: one layer's trainable vision-branch groups in the pack.
type motSpans struct {
	inputNorm, q, k, v, o, postNorm, gate, up, down int
}

// motFrozen: one layer's frozen graph participants.
type motFrozen struct {
	textInputNorm, textPostNorm []float32
	textQ, textK, textV, textO  BF16Matrix
	textGate, textUp, textDown  BF16Matrix
	qNorm, kNorm                [2][]float32
}

// ModalityTransformerTrainer: the packed vision fork of every layer over the
// shared Muon stepper, with the frozen text branch alongside for the real
// branch-routed forward.
type ModalityTransformerTrainer struct {
	cfg       Config
	rope      RopePlan
	frozen    []motFrozen
	spans     []motSpans
	finalNorm []float32
	head      []uint16 // tied head, BF16 [vocab, hidden]
	pack      *optimizer.TensorPack
	stepper   optimizer.Stepper
	config    optimizer.Config
	step      int
}

// NewModalityTransformerTrainer packs the vision branch of each layer as f32
// masters and compiles the Muon plan. Base LR derives from the packed
// parameter count; momentum from the CLT effective-samples rule. The trainer
// takes ownership of the layer weights (vision BF16 storage is released
// after master decode).
func NewModalityTransformerTrainer(cfg Config, layers []LayerWeights, finalNorm []float32, head []uint16, policy trainingprogram.OptimizerPolicy) (*ModalityTransformerTrainer, error) {
	if len(layers) != cfg.NumHiddenLayers || cfg.NumHiddenLayers == 0 {
		return nil, fmt.Errorf("routed lm mot train: %d layers, config wants %d", len(layers), cfg.NumHiddenLayers)
	}
	d, inter := cfg.HiddenSize, cfg.IntermediateSize
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	if len(finalNorm) != d || len(head) == 0 || len(head)%d != 0 {
		return nil, fmt.Errorf("routed lm mot train: final norm len=%d head len=%d, hidden %d", len(finalNorm), len(head), d)
	}
	rope, err := ropePlanFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	t := &ModalityTransformerTrainer{
		cfg: cfg, rope: rope,
		frozen:    make([]motFrozen, len(layers)),
		spans:     make([]motSpans, len(layers)),
		finalNorm: finalNorm, head: head,
	}
	var sections []optimizer.TensorSection
	for layer := range layers {
		w := &layers[layer]
		sp := &t.spans[layer]
		for _, target := range []struct {
			group   *int
			section optimizer.TensorSection
		}{
			{&sp.inputNorm, optimizer.TensorSection{Name: suffixInputNorm, Rows: 1, Cols: d, Values: w.InputNorm.Vision}},
			{&sp.q, optimizer.TensorSection{Name: suffixQProj, Rows: qOut, Cols: d, BF16: w.QKV.QVision.Data}},
			{&sp.k, optimizer.TensorSection{Name: suffixKProj, Rows: kvOut, Cols: d, BF16: w.QKV.KVision.Data}},
			{&sp.v, optimizer.TensorSection{Name: suffixVProj, Rows: kvOut, Cols: d, BF16: w.QKV.VVision.Data}},
			{&sp.o, optimizer.TensorSection{Name: suffixOProj, Rows: d, Cols: qOut, BF16: w.QKV.OVision.Data}},
			{&sp.postNorm, optimizer.TensorSection{Name: suffixPostNorm, Rows: 1, Cols: d, Values: w.Output.PostVision}},
			{&sp.gate, optimizer.TensorSection{Name: suffixGateProj, Rows: inter, Cols: d, BF16: w.Output.GateVision.Data}},
			{&sp.up, optimizer.TensorSection{Name: suffixUpProj, Rows: inter, Cols: d, BF16: w.Output.UpVision.Data}},
			{&sp.down, optimizer.TensorSection{Name: suffixDownProj, Rows: d, Cols: inter, BF16: w.Output.DownVision.Data}},
		} {
			*target.group = len(sections)
			target.section.Name = fmt.Sprintf("layers.%d.vision.%s", layer, target.section.Name)
			sections = append(sections, target.section)
		}
		t.frozen[layer] = motFrozen{
			textInputNorm: w.InputNorm.Text, textPostNorm: w.Output.PostText,
			textQ: w.QKV.QText, textK: w.QKV.KText, textV: w.QKV.VText, textO: w.QKV.OText,
			textGate: w.Output.GateText, textUp: w.Output.UpText, textDown: w.Output.DownText,
			qNorm: w.QKV.QNorm, kNorm: w.QKV.KNorm,
		}
		// Release the vision BF16 storage — the masters own it once packed.
		w.QKV.QVision, w.QKV.KVision, w.QKV.VVision, w.QKV.OVision = BF16Matrix{}, BF16Matrix{}, BF16Matrix{}, BF16Matrix{}
		w.Output.GateVision, w.Output.UpVision, w.Output.DownVision = BF16Matrix{}, BF16Matrix{}, BF16Matrix{}
	}
	if t.pack, err = optimizer.NewSectionPack(sections); err != nil {
		return nil, fmt.Errorf("routed lm mot train: %w", err)
	}
	if t.config, err = policy.Config(t.pack.ParameterCount()); err != nil {
		return nil, err
	}
	if t.stepper, err = t.pack.NewStepper(t.config); err != nil {
		return nil, err
	}
	return t, nil
}

// ParameterCount reports the packed trainable parameter total.
func (t *ModalityTransformerTrainer) ParameterCount() int { return t.pack.ParameterCount() }

// Config exposes the derived hyperparameters for evidence.
func (t *ModalityTransformerTrainer) Config() optimizer.Config { return t.config }

// OptimizerPlan is the compiled Muon plan over the packed vision fork: one
// group per layer tensor.
func (t *ModalityTransformerTrainer) OptimizerPlan() optimizer.Plan { return t.pack.Plan() }

// Snapshot copies the optimizer state the next step continues from.
func (t *ModalityTransformerTrainer) Snapshot() optimizer.State { return t.stepper.Snapshot() }

// SaveWeights writes the trained vision fork at path, the weights Snapshot
// pairs with.
func (t *ModalityTransformerTrainer) SaveWeights(path string) error { return t.pack.Save(path) }

// Close releases the stepper backend.
func (t *ModalityTransformerTrainer) Close() error { return t.stepper.Close() }

func (t *ModalityTransformerTrainer) view(group int) []float32     { return t.pack.Weights(group) }
func (t *ModalityTransformerTrainer) gradView(group int) []float32 { return t.pack.Gradients(group) }

// motActivations: one layer's retained training-precision intermediates.
type motActivations struct {
	input          []float32 // [t, d] layer input
	normRows       []float32 // [t, d]
	qProj, kProj   []float32 // pre-rope [t, qOut] / [t, kvOut]
	vProj          []float32 // [t, kvOut]
	qRoped, kRoped []float32 // post-rope, pre-QK-norm
	qHeads, kHeads []float32 // post-QK-norm
	context        []float32 // [t, qOut]
	residual       []float32 // input + o-projection [t, d]
	postNorm       []float32 // [t, d]
	gatePre, upPre []float32 // [t, inter] pre-activation
	output         []float32 // [t, d]
}

// Loss runs the training-precision forward and returns the mean next-token
// cross-entropy at the supervised positions without touching gradients.
func (t *ModalityTransformerTrainer) Loss(hidden []float32, mask []int, targets []MoTTarget) (float64, error) {
	activations, err := t.forward(hidden, mask)
	if err != nil {
		return 0, err
	}
	loss, _, _, err := t.terminalLoss(activations[len(activations)-1].output, len(mask), targets, false)
	return loss, err
}

// Step: one observed full-pipeline training step — forward through every
// branch-routed layer, next-token cross-entropy at the supervised positions,
// full VJP into the packed vision masters, Muon update.
func (t *ModalityTransformerTrainer) Step(hidden []float32, mask []int, targets []MoTTarget) (MoTTrainStepResult, error) {
	loss, gradientL2, err := t.lossAndGradients(hidden, mask, targets)
	if err != nil {
		return MoTTrainStepResult{}, err
	}
	return optimizer.Advance(t.stepper, t.config, &t.step, loss, gradientL2)
}

func (t *ModalityTransformerTrainer) lossAndGradients(hidden []float32, mask []int, targets []MoTTarget) (float64, float64, error) {
	activations, err := t.forward(hidden, mask)
	if err != nil {
		return 0, 0, err
	}
	tokens := len(mask)
	loss, dOutput, _, err := t.terminalLoss(activations[len(activations)-1].output, tokens, targets, true)
	if err != nil {
		return 0, 0, err
	}
	t.pack.ClearGradients()
	for layer := len(activations) - 1; layer >= 0; layer-- {
		dOutput = t.layerBackward(layer, activations[layer], mask, dOutput)
	}
	return loss, t.pack.GradientL2(), nil
}

// forward: training-precision (f32, unrounded) mirror of PromptLayerForward
// over every layer, retaining the intermediates the backward consumes.
func (t *ModalityTransformerTrainer) forward(hidden []float32, mask []int) ([]motActivations, error) {
	tokens := len(mask)
	d := t.cfg.HiddenSize
	if tokens == 0 || len(hidden) != tokens*d {
		return nil, fmt.Errorf("routed lm mot train: hidden len %d != %d tokens * %d", len(hidden), tokens, d)
	}
	activations := make([]motActivations, len(t.frozen))
	input := hidden
	for layer := range t.frozen {
		a, err := t.layerForward(layer, input, mask)
		if err != nil {
			return nil, err
		}
		activations[layer] = a
		input = a.output
	}
	return activations, nil
}

// branchLinear: rows of one branch through a frozen BF16 or trainable f32
// weight — pack, project, scatter.
func packRows(dst, src []float32, rows []int, width int) {
	for i, token := range rows {
		copy(dst[i*width:(i+1)*width], src[token*width:(token+1)*width])
	}
}

func scatterRows(dst, src []float32, rows []int, width int) {
	for i, token := range rows {
		copy(dst[token*width:(token+1)*width], src[i*width:(i+1)*width])
	}
}

func (t *ModalityTransformerTrainer) layerForward(layer int, input []float32, mask []int) (motActivations, error) {
	cfg := t.cfg
	d, inter := cfg.HiddenSize, cfg.IntermediateSize
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	hd := cfg.HeadDim
	tokens := len(mask)
	frozen := t.frozen[layer]
	sp := t.spans[layer]
	a := motActivations{
		input:    input,
		normRows: make([]float32, tokens*d),
		qProj:    make([]float32, tokens*qOut),
		kProj:    make([]float32, tokens*kvOut),
		vProj:    make([]float32, tokens*kvOut),
		context:  make([]float32, tokens*qOut),
		residual: make([]float32, tokens*d),
		postNorm: make([]float32, tokens*d),
		gatePre:  make([]float32, tokens*inter),
		upPre:    make([]float32, tokens*inter),
		output:   make([]float32, tokens*d),
	}
	zero, nonZero := branchRows(mask)
	// Pre-attention norm, branch-selected scale (unrounded).
	for _, token := range zero {
		hostmath.RMSNormInto(a.normRows[token*d:(token+1)*d], input[token*d:(token+1)*d], frozen.textInputNorm, 1, d, cfg.RMSNormEps)
	}
	for _, token := range nonZero {
		hostmath.RMSNormInto(a.normRows[token*d:(token+1)*d], input[token*d:(token+1)*d], t.view(sp.inputNorm), 1, d, cfg.RMSNormEps)
	}
	// Branch-batched projections: text rows through frozen BF16, vision rows
	// through the f32 masters.
	project := func(rows []int, frozenW [3]BF16Matrix, master [3]int) {
		if len(rows) == 0 {
			return
		}
		pack := make([]float32, len(rows)*d)
		packRows(pack, a.normRows, rows, d)
		outs := [3]int{qOut, kvOut, kvOut}
		dsts := [3][]float32{a.qProj, a.kProj, a.vProj}
		for i := range 3 {
			projected := make([]float32, len(rows)*outs[i])
			if frozenW[i].Data != nil {
				hostmath.LinearBF16(projected, pack, frozenW[i].Data, len(rows), d, outs[i])
			} else {
				hostmath.Linear(projected, pack, t.view(master[i]), len(rows), d, outs[i])
			}
			scatterRows(dsts[i], projected, rows, outs[i])
		}
	}
	project(zero, [3]BF16Matrix{frozen.textQ, frozen.textK, frozen.textV}, [3]int{})
	project(nonZero, [3]BF16Matrix{}, [3]int{sp.q, sp.k, sp.v})
	// Rope (unrounded) + per-head QK-norm (frozen scales).
	a.qRoped = append([]float32(nil), a.qProj...)
	a.kRoped = append([]float32(nil), a.kProj...)
	for token := range tokens {
		pos := RowPosition{Branch: branchIndex(mask[token]), Time: token}
		for head := 0; head < cfg.NumAttentionHeads; head++ {
			t.rope.applyRotaryF32(a.qRoped[token*qOut+head*hd:token*qOut+(head+1)*hd], pos, false)
		}
		for head := 0; head < cfg.NumKeyValueHeads; head++ {
			t.rope.applyRotaryF32(a.kRoped[token*kvOut+head*hd:token*kvOut+(head+1)*hd], pos, false)
		}
	}
	a.qHeads = make([]float32, len(a.qRoped))
	a.kHeads = make([]float32, len(a.kRoped))
	for token := range tokens {
		branch := branchIndex(mask[token])
		for head := 0; head < cfg.NumAttentionHeads; head++ {
			offset := token*qOut + head*hd
			hostmath.RMSNormInto(a.qHeads[offset:offset+hd], a.qRoped[offset:offset+hd], frozen.qNorm[branch], 1, hd, cfg.RMSNormEps)
		}
		for head := 0; head < cfg.NumKeyValueHeads; head++ {
			offset := token*kvOut + head*hd
			hostmath.RMSNormInto(a.kHeads[offset:offset+hd], a.kRoped[offset:offset+hd], frozen.kNorm[branch], 1, hd, cfg.RMSNormEps)
		}
	}
	// Window-masked attention (per-row windows from the visual segments).
	segments := VisualSegments(mask)
	windows := SegmentWindows(segments, tokens)
	group := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	scale := 1.0 / math.Sqrt(float64(hd))
	hostmath.ParallelRangeF64(tokens, cfg.NumAttentionHeads*tokens*2*hd, func(lo, hi int) {
		scores := make([]float32, tokens)
		for tokenPos := lo; tokenPos < hi; tokenPos++ {
			start, end := windows[tokenPos][0], windows[tokenPos][1]
			for head := 0; head < cfg.NumAttentionHeads; head++ {
				kvHead := head / group
				q := a.qHeads[tokenPos*qOut+head*hd : tokenPos*qOut+(head+1)*hd]
				window := scores[:end-start]
				for keyPos := start; keyPos < end; keyPos++ {
					k := a.kHeads[keyPos*kvOut+kvHead*hd : keyPos*kvOut+(kvHead+1)*hd]
					var dot float64
					for i := range hd {
						dot += float64(q[i]) * float64(k[i])
					}
					window[keyPos-start] = float32(dot * scale)
				}
				hostmath.SoftmaxInPlace(window)
				out := a.context[tokenPos*qOut+head*hd : tokenPos*qOut+(head+1)*hd]
				for keyPos, prob := range window {
					v := a.vProj[(start+keyPos)*kvOut+kvHead*hd : (start+keyPos)*kvOut+(kvHead+1)*hd]
					for i := range hd {
						out[i] += prob * v[i]
					}
				}
			}
		}
	})
	// Output projection, residual, post-norm, branch MLP.
	body := func(rows []int, oText BF16Matrix, oMaster int, postWeight []float32, gateText, upText, downText BF16Matrix, gateM, upM, downM int) {
		if len(rows) == 0 {
			return
		}
		pack := make([]float32, len(rows)*d)
		packRows(pack, a.context, rows, qOut)
		projected := make([]float32, len(rows)*d)
		if oText.Data != nil {
			hostmath.LinearBF16(projected, pack, oText.Data, len(rows), qOut, d)
		} else {
			hostmath.Linear(projected, pack, t.view(oMaster), len(rows), qOut, d)
		}
		for i, token := range rows {
			residualRow := a.residual[token*d : (token+1)*d]
			inputRow := a.input[token*d : (token+1)*d]
			projectedRow := projected[i*d : (i+1)*d]
			for c := range d {
				residualRow[c] = inputRow[c] + projectedRow[c]
			}
			hostmath.RMSNormInto(a.postNorm[token*d:(token+1)*d], residualRow, postWeight, 1, d, cfg.RMSNormEps)
		}
		groupPost := make([]float32, len(rows)*d)
		packRows(groupPost, a.postNorm, rows, d)
		gateValues := make([]float32, len(rows)*inter)
		upValues := make([]float32, len(rows)*inter)
		if gateText.Data != nil {
			hostmath.LinearBF16(gateValues, groupPost, gateText.Data, len(rows), d, inter)
			hostmath.LinearBF16(upValues, groupPost, upText.Data, len(rows), d, inter)
		} else {
			hostmath.Linear(gateValues, groupPost, t.view(gateM), len(rows), d, inter)
			hostmath.Linear(upValues, groupPost, t.view(upM), len(rows), d, inter)
		}
		scatterRows(a.gatePre, gateValues, rows, inter)
		scatterRows(a.upPre, upValues, rows, inter)
		activated := make([]float32, len(gateValues))
		for i := range gateValues {
			activated[i] = float32(silu64(float64(gateValues[i]))) * upValues[i]
		}
		downValues := make([]float32, len(rows)*d)
		if downText.Data != nil {
			hostmath.LinearBF16(downValues, activated, downText.Data, len(rows), inter, d)
		} else {
			hostmath.Linear(downValues, activated, t.view(downM), len(rows), inter, d)
		}
		for i, token := range rows {
			outRow := a.output[token*d : (token+1)*d]
			residualRow := a.residual[token*d : (token+1)*d]
			downRow := downValues[i*d : (i+1)*d]
			for c := range d {
				outRow[c] = residualRow[c] + downRow[c]
			}
		}
	}
	body(zero, frozen.textO, 0, frozen.textPostNorm, frozen.textGate, frozen.textUp, frozen.textDown, 0, 0, 0)
	body(nonZero, BF16Matrix{}, sp.o, t.view(sp.postNorm), BF16Matrix{}, BF16Matrix{}, BF16Matrix{}, sp.gate, sp.up, sp.down)
	return a, nil
}

// layerBackward: full VJP of layerForward. Vision-row weight gradients
// accumulate into the packed masters; text-row weights are frozen graph
// participants (gradient flows through, no weight gradient). Returns the
// gradient at the layer input.
func (t *ModalityTransformerTrainer) layerBackward(layer int, a motActivations, mask []int, dOutput []float32) []float32 {
	cfg := t.cfg
	d, inter := cfg.HiddenSize, cfg.IntermediateSize
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	hd := cfg.HeadDim
	tokens := len(mask)
	frozen := t.frozen[layer]
	sp := t.spans[layer]
	zero, nonZero := branchRows(mask)

	dResidual := make([]float32, tokens*d)
	dContext := make([]float32, tokens*qOut)
	// MLP + residual + post-norm + o-projection, per branch.
	mlpBackward := func(rows []int, trainable bool) {
		if len(rows) == 0 {
			return
		}
		n := len(rows)
		// Recompute activated = silu(gate) * up for the down input.
		gatePack := make([]float32, n*inter)
		upPack := make([]float32, n*inter)
		packRows(gatePack, a.gatePre, rows, inter)
		packRows(upPack, a.upPre, rows, inter)
		activated := make([]float32, n*inter)
		for i := range activated {
			activated[i] = float32(silu64(float64(gatePack[i]))) * upPack[i]
		}
		dOut := make([]float32, n*d)
		packRows(dOut, dOutput, rows, d)
		// output = residual + down(activated): dDown = dOut, dResidual += dOut.
		dActivated := make([]float32, n*inter)
		if trainable {
			hostmath.Linear(dActivated, dOut, hostmath.Transpose2D(t.view(sp.down), d, inter), n, d, inter)
			linearWeightGradient(t.gradView(sp.down), activated, dOut, n, inter, d)
		} else {
			hostmath.LinearBF16BackwardInput(dActivated, dOut, frozen.textDown.Data, n, inter, d)
		}
		dGate := make([]float32, n*inter)
		dUp := make([]float32, n*inter)
		hostmath.SiLUGateBackward(dGate, dUp, gatePack, upPack, dActivated)
		dPostNorm := make([]float32, n*d)
		if trainable {
			hostmath.Linear(dPostNorm, dGate, hostmath.Transpose2D(t.view(sp.gate), inter, d), n, inter, d)
			linearWeightGradient(t.gradView(sp.gate), packScratch(a.postNorm, rows, d), dGate, n, d, inter)
			dUpX := make([]float32, n*d)
			hostmath.Linear(dUpX, dUp, hostmath.Transpose2D(t.view(sp.up), inter, d), n, inter, d)
			for i := range dPostNorm {
				dPostNorm[i] += dUpX[i]
			}
			linearWeightGradient(t.gradView(sp.up), packScratch(a.postNorm, rows, d), dUp, n, d, inter)
		} else {
			hostmath.LinearBF16BackwardInput(dPostNorm, dGate, frozen.textGate.Data, n, d, inter)
			dUpX := make([]float32, n*d)
			hostmath.LinearBF16BackwardInput(dUpX, dUp, frozen.textUp.Data, n, d, inter)
			for i := range dPostNorm {
				dPostNorm[i] += dUpX[i]
			}
		}
		// post-norm backward: x = residual rows.
		residualPack := packScratch(a.residual, rows, d)
		dResidualPack := make([]float32, n*d)
		if trainable {
			hostmath.RMSNormBackward(dResidualPack, t.gradView(sp.postNorm), residualPack, t.view(sp.postNorm), dPostNorm, n, d, cfg.RMSNormEps, false)
		} else {
			hostmath.RMSNormBackward(dResidualPack, nil, residualPack, frozen.textPostNorm, dPostNorm, n, d, cfg.RMSNormEps, false)
		}
		// dResidual = dOut (residual add) + norm path.
		for i := range dResidualPack {
			dResidualPack[i] += dOut[i]
		}
		scatterRows(dResidual, dResidualPack, rows, d)
		// o-projection backward: attnProjected = o(context); residual = input + attnProjected.
		contextPack := packScratch(a.context, rows, qOut)
		dContextPack := make([]float32, n*qOut)
		if trainable {
			hostmath.Linear(dContextPack, dResidualPack, hostmath.Transpose2D(t.view(sp.o), d, qOut), n, d, qOut)
			linearWeightGradient(t.gradView(sp.o), contextPack, dResidualPack, n, qOut, d)
		} else {
			hostmath.LinearBF16BackwardInput(dContextPack, dResidualPack, frozen.textO.Data, n, qOut, d)
		}
		scatterRows(dContext, dContextPack, rows, qOut)
	}
	mlpBackward(zero, false)
	mlpBackward(nonZero, true)

	// Attention backward: recompute window probabilities, accumulate
	// dQHeads/dKHeads/dVProj (parallel over query tokens; key/value gradients
	// merge from worker-local buffers).
	segments := VisualSegments(mask)
	windows := SegmentWindows(segments, tokens)
	group := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	scale := 1.0 / math.Sqrt(float64(hd))
	dQHeads := make([]float32, tokens*qOut)
	dKHeads := make([]float32, tokens*kvOut)
	dVProj := make([]float32, tokens*kvOut)
	var mergeMu sync.Mutex
	hostmath.ParallelRangeF64(tokens, cfg.NumAttentionHeads*tokens*4*hd, func(lo, hi int) {
		localDK := make([]float32, tokens*kvOut)
		localDV := make([]float32, tokens*kvOut)
		scores := make([]float32, tokens)
		dProbs := make([]float64, tokens)
		for tokenPos := lo; tokenPos < hi; tokenPos++ {
			start, end := windows[tokenPos][0], windows[tokenPos][1]
			for head := 0; head < cfg.NumAttentionHeads; head++ {
				kvHead := head / group
				q := a.qHeads[tokenPos*qOut+head*hd : tokenPos*qOut+(head+1)*hd]
				window := scores[:end-start]
				for keyPos := start; keyPos < end; keyPos++ {
					k := a.kHeads[keyPos*kvOut+kvHead*hd : keyPos*kvOut+(kvHead+1)*hd]
					var dot float64
					for i := range hd {
						dot += float64(q[i]) * float64(k[i])
					}
					window[keyPos-start] = float32(dot * scale)
				}
				hostmath.SoftmaxInPlace(window)
				dCtx := dContext[tokenPos*qOut+head*hd : tokenPos*qOut+(head+1)*hd]
				// dV[key] += p * dCtx ; dP[key] = dCtx · V[key]
				var dotPdP float64
				dP := dProbs[:end-start]
				for keyPos := start; keyPos < end; keyPos++ {
					v := a.vProj[keyPos*kvOut+kvHead*hd : keyPos*kvOut+(kvHead+1)*hd]
					dv := localDV[keyPos*kvOut+kvHead*hd : keyPos*kvOut+(kvHead+1)*hd]
					prob := window[keyPos-start]
					var dot float64
					for i := range hd {
						dv[i] += prob * dCtx[i]
						dot += float64(dCtx[i]) * float64(v[i])
					}
					dP[keyPos-start] = dot
					dotPdP += float64(prob) * dot
				}
				// softmax VJP: dS = p * (dP - p·dP), then dQ/dK through the dot.
				dq := dQHeads[tokenPos*qOut+head*hd : tokenPos*qOut+(head+1)*hd]
				for keyPos := start; keyPos < end; keyPos++ {
					prob := float64(window[keyPos-start])
					dScore := prob * (dP[keyPos-start] - dotPdP) * scale
					if dScore == 0 {
						continue
					}
					k := a.kHeads[keyPos*kvOut+kvHead*hd : keyPos*kvOut+(kvHead+1)*hd]
					dk := localDK[keyPos*kvOut+kvHead*hd : keyPos*kvOut+(kvHead+1)*hd]
					for i := range hd {
						dq[i] += float32(dScore * float64(k[i]))
						dk[i] += float32(dScore * float64(q[i]))
					}
				}
			}
		}
		mergeMu.Lock()
		for i, v := range localDK {
			dKHeads[i] += v
		}
		for i, v := range localDV {
			dVProj[i] += v
		}
		mergeMu.Unlock()
	})

	// QK-norm backward (frozen scales) + inverse rope back to the projections.
	dQProj := make([]float32, tokens*qOut)
	dKProj := make([]float32, tokens*kvOut)
	for token := range tokens {
		branch := branchIndex(mask[token])
		pos := RowPosition{Branch: branch, Time: token}
		for head := 0; head < cfg.NumAttentionHeads; head++ {
			offset := token*qOut + head*hd
			hostmath.RMSNormBackward(dQProj[offset:offset+hd], nil, a.qRoped[offset:offset+hd], frozen.qNorm[branch], dQHeads[offset:offset+hd], 1, hd, cfg.RMSNormEps, false)
			t.rope.applyRotaryF32(dQProj[offset:offset+hd], pos, true)
		}
		for head := 0; head < cfg.NumKeyValueHeads; head++ {
			offset := token*kvOut + head*hd
			hostmath.RMSNormBackward(dKProj[offset:offset+hd], nil, a.kRoped[offset:offset+hd], frozen.kNorm[branch], dKHeads[offset:offset+hd], 1, hd, cfg.RMSNormEps, false)
			t.rope.applyRotaryF32(dKProj[offset:offset+hd], pos, true)
		}
	}

	// Q/K/V projection backward into dNormRows, then input-norm backward.
	dNormRows := make([]float32, tokens*d)
	dInput := make([]float32, tokens*d)
	projBackward := func(rows []int, trainable bool) {
		if len(rows) == 0 {
			return
		}
		n := len(rows)
		normPack := packScratch(a.normRows, rows, d)
		dNormPack := make([]float32, n*d)
		sources := []struct {
			grad    []float32
			out     int
			frozenW BF16Matrix
			master  int
		}{
			{dQProj, qOut, frozen.textQ, sp.q},
			{dKProj, kvOut, frozen.textK, sp.k},
			{dVProj, kvOut, frozen.textV, sp.v},
		}
		for _, source := range sources {
			dyPack := make([]float32, n*source.out)
			packRows(dyPack, source.grad, rows, source.out)
			dx := make([]float32, n*d)
			if trainable {
				hostmath.Linear(dx, dyPack, hostmath.Transpose2D(t.view(source.master), source.out, d), n, source.out, d)
				linearWeightGradient(t.gradView(source.master), normPack, dyPack, n, d, source.out)
			} else {
				hostmath.LinearBF16BackwardInput(dx, dyPack, source.frozenW.Data, n, d, source.out)
			}
			for i := range dNormPack {
				dNormPack[i] += dx[i]
			}
		}
		scatterRows(dNormRows, dNormPack, rows, d)
		inputPack := packScratch(a.input, rows, d)
		dInputPack := make([]float32, n*d)
		if trainable {
			hostmath.RMSNormBackward(dInputPack, t.gradView(sp.inputNorm), inputPack, t.view(sp.inputNorm), dNormPack, n, d, cfg.RMSNormEps, false)
		} else {
			hostmath.RMSNormBackward(dInputPack, nil, inputPack, frozen.textInputNorm, dNormPack, n, d, cfg.RMSNormEps, false)
		}
		scatterRows(dInput, dInputPack, rows, d)
	}
	projBackward(zero, false)
	projBackward(nonZero, true)

	// dInput also carries the residual path (residual = input + attnProjected).
	for i := range dInput {
		dInput[i] += dResidual[i]
	}
	return dInput
}

// terminalLoss: final norm + tied-head cross-entropy at the supervised
// positions. withGradient additionally returns the gradient at the last
// layer's output rows.
func (t *ModalityTransformerTrainer) terminalLoss(output []float32, tokens int, targets []MoTTarget, withGradient bool) (float64, []float32, []float32, error) {
	d := t.cfg.HiddenSize
	vocab := len(t.head) / d
	if len(targets) == 0 {
		return 0, nil, nil, fmt.Errorf("routed lm mot train: no supervised positions")
	}
	finalHidden := make([]float32, tokens*d)
	hostmath.RMSNormInto(finalHidden, output, t.finalNorm, tokens, d, t.cfg.RMSNormEps)
	n := len(targets)
	finalRows := make([]float32, n*d)
	for i, target := range targets {
		if target.Position < 0 || target.Position >= tokens || target.Token < 0 || target.Token >= vocab {
			return 0, nil, nil, fmt.Errorf("routed lm mot train: target %+v outside prompt %d / vocab %d", target, tokens, vocab)
		}
		copy(finalRows[i*d:(i+1)*d], finalHidden[target.Position*d:(target.Position+1)*d])
	}
	logits := make([]float32, n*vocab)
	hostmath.LinearBF16(logits, finalRows, t.head, n, d, vocab)
	invN := 1 / float64(n)
	var loss float64
	for i, target := range targets {
		row := logits[i*vocab : (i+1)*vocab]
		maxLogit := slices.Max(row)
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v - maxLogit))
		}
		loss += (math.Log(sum) - float64(row[target.Token]-maxLogit)) * invN
		if withGradient {
			// row becomes dLogits in place: (softmax - onehot)/n.
			for j := range row {
				row[j] = float32(math.Exp(float64(row[j]-maxLogit)) / sum * invN)
			}
			row[target.Token] -= float32(invN)
		}
	}
	if !withGradient {
		return loss, nil, finalHidden, nil
	}
	dFinalRows := make([]float32, n*d)
	hostmath.LinearBF16BackwardInput(dFinalRows, logits, t.head, n, d, vocab)
	dFinalHidden := make([]float32, tokens*d)
	for i, target := range targets {
		row := dFinalHidden[target.Position*d : (target.Position+1)*d]
		for c := range d {
			row[c] += dFinalRows[i*d+c]
		}
	}
	dOutput := make([]float32, tokens*d)
	hostmath.RMSNormBackward(dOutput, nil, output, t.finalNorm, dFinalHidden, tokens, d, t.cfg.RMSNormEps, false)
	return loss, dOutput, finalHidden, nil
}

// applyRotaryF32: unrounded per-section rotate-half (training precision);
// invert applies the transpose (exact inverse of the orthogonal rotation).
func (p RopePlan) applyRotaryF32(row []float32, pos RowPosition, invert bool) {
	offset := 0
	for i, section := range p.Sections {
		span := row[offset : offset+section.Width]
		half := section.Width / 2
		position := float64(pos.axis(section.Axis))
		for j := range half {
			angle := position * p.invFreq[i][j]
			c, s := math.Cos(angle), math.Sin(angle)
			if invert {
				s = -s
			}
			a, b := float64(span[j]), float64(span[j+half])
			span[j] = float32(a*c - b*s)
			span[j+half] = float32(b*c + a*s)
		}
		offset += section.Width
	}
}

// linearWeightGradient: the shared parallel dW += dy^T ⊗ x kernel.
func linearWeightGradient(dW, x, dy []float32, rows, inDim, outDim int) {
	hostmath.LinearWeightGradient(dW, x, dy, rows, inDim, outDim)
}

// packScratch: gather branch rows into a fresh packed block.
func packScratch(src []float32, rows []int, width int) []float32 {
	pack := make([]float32, len(rows)*width)
	packRows(pack, src, rows, width)
	return pack
}
