// Full-DiT trainer for the conditioned video diffusion transformer: EVERY
// checkpoint tensor — all transformer blocks, the patch embedding, the text
// and time embeddings, the time projection, and the modulated head — trains
// as f32 masters over the shared Muon stepper with fully derived
// hyperparameters. The forward is the training-precision (f64-accumulated,
// unrounded) host mirror of the serving graph
// (model.buildConditionedDiffusionBlock): adaptive-layernorm modulation with
// (shift, scale, gate) chunk order, full-width pre-head-split QK RMS norms,
// 3-axis adjacent-pair rotary, fixed-context cross-attention, and the
// tanh-GELU feed-forward. The objective is real flow matching — MSE against
// v = noise - x0 on the shifted flow schedule. The frozen text encoder and
// VAE sit outside this surface; nothing inside the DiT is frozen. This is
// the "full video-diffusion training" promotion-gate trainer.
package latentvideo

import (
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/model"
	"overgo/internal/optimizer"
	"overgo/internal/pytorchzip"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/trainingprogram"
)

// DiTTrainStepResult: measured facts of one observed full-DiT training step.
type DiTTrainStepResult = optimizer.ObservedStepResult

// DiTTrainBatch: one real training example — the (possibly source-extended)
// noised latent, the raw frozen-encoder text rows, the flow timestep, and
// the velocity target.
type DiTTrainBatch struct {
	Latent     []float32 // [InDim, latentFrames, latentHeight, latentWidth] channel-major
	RawText    []float32 // [TextTokens, textDim] compacted real encoder rows
	TextTokens int
	Timestep   float64
	Target     []float32 // [OutDim, latentFrames, latentHeight, latentWidth] velocity target
}

type ditSpan struct{ start, end int }

// DiTTrainer: the packed full-DiT tensor set over the shared Muon stepper.
type DiTTrainer struct {
	cfg      DenoiserConfig
	textDim  int
	geometry LatentGeometry

	spans     map[string]ditSpan
	weights   []float32
	gradients []float32

	axisChannels [3]int
	invFreq      [3][]float64
	positions    [][3]int

	// currentContext: the projected context rows the running
	// forward/backward pair consumes (cross K/V linear inputs).
	currentContext []float32

	stepper optimizer.Stepper
	optCfg  optimizer.Config
	step    int
}

type ditTensorSpec struct {
	name       string
	rows, cols int
}

// ditTensorSpecs: the complete trainable tensor contract in packing order —
// prologue (patch embedding, text embedding, time embedding/projection,
// head) then every block. Names are the checkpoint tensor names.
func ditTensorSpecs(c DenoiserConfig, textDim int) []ditTensorSpec {
	d, f := c.Dim, c.FFNDim
	one := tensor.SingletonExtent
	blockWidth := media.PairedShiftScaleGateWidth(d)
	headModulationWidth := tensor.PairedExtent * d
	specs := []ditTensorSpec{
		{"patch_embedding.weight", d, c.patchIn()},
		{"patch_embedding.bias", one, d},
		{"text_embedding.0.weight", d, textDim},
		{"text_embedding.0.bias", one, d},
		{"text_embedding.2.weight", d, d},
		{"text_embedding.2.bias", one, d},
		{"time_embedding.0.weight", d, c.FreqDim},
		{"time_embedding.0.bias", one, d},
		{"time_embedding.2.weight", d, d},
		{"time_embedding.2.bias", one, d},
		{"time_projection.1.weight", blockWidth, d},
		{"time_projection.1.bias", one, blockWidth},
		{"head.modulation", one, headModulationWidth},
		{"head.head.weight", c.patchOut(), d},
		{"head.head.bias", one, c.patchOut()},
	}
	for layer := range c.NumLayers {
		prefix := denoiserBlockPrefix(layer)
		for _, attention := range []string{"self_attn.", "cross_attn."} {
			for _, projection := range []string{"q", "k", "v", "o"} {
				specs = append(specs,
					ditTensorSpec{prefix + attention + projection + ".weight", d, d},
					ditTensorSpec{prefix + attention + projection + ".bias", one, d},
				)
			}
			specs = append(specs,
				ditTensorSpec{prefix + attention + "norm_q.weight", one, d},
				ditTensorSpec{prefix + attention + "norm_k.weight", one, d},
			)
		}
		specs = append(specs,
			ditTensorSpec{prefix + "ffn.0.weight", f, d},
			ditTensorSpec{prefix + "ffn.0.bias", one, f},
			ditTensorSpec{prefix + "ffn.2.weight", d, f},
			ditTensorSpec{prefix + "ffn.2.bias", one, d},
			ditTensorSpec{prefix + "modulation", one, blockWidth},
			ditTensorSpec{prefix + "norm3.weight", one, d},
			ditTensorSpec{prefix + "norm3.bias", one, d},
		)
	}
	return specs
}

// NewDiTTrainer packs every trainable tensor as f32 masters and compiles the
// Muon plan. Base LR derives from the packed parameter count; momentum from
// the CLT effective-samples rule. The trainer takes ownership of the loaded
// tensor map, releasing each source slice as it packs (the map is emptied).
func NewDiTTrainer(cfg DenoiserConfig, textDim int, geometry LatentGeometry, tensors map[string][]float32) (*DiTTrainer, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if !checked.PositiveInts(textDim) {
		return nil, fmt.Errorf("dit train: text dim %d must be positive", textDim)
	}
	volume := media.VolumeGeometry{Frames: geometry.LatentFrames, Height: geometry.LatentHeight, Width: geometry.LatentWidth}
	grid, err := media.VolumePatchGrid(volume, cfg.PatchSize)
	if err != nil {
		return nil, fmt.Errorf("dit train: geometry: %w", err)
	}
	sequence, err := media.VolumeElements(grid)
	if err != nil {
		return nil, fmt.Errorf("dit train: geometry: %w", err)
	}
	if !checked.Equal(geometry.Channels, cfg.InDim) {
		return nil, fmt.Errorf("dit train: geometry %+v incompatible with config", geometry)
	}
	if !checked.Equal(geometry.Grid, grid) {
		return nil, fmt.Errorf("dit train: geometry %+v incompatible with config", geometry)
	}
	if !checked.Equal(geometry.Seq, sequence) {
		return nil, fmt.Errorf("dit train: geometry %+v incompatible with config", geometry)
	}
	headWidth := cfg.Dim / cfg.NumHeads
	channels := model.ThreeAxisRotaryChannels(uint64(headWidth))
	trainer := &DiTTrainer{
		cfg: cfg, textDim: textDim, geometry: geometry,
		spans:        make(map[string]ditSpan),
		axisChannels: [3]int{int(channels[0]), int(channels[1]), int(channels[2])},
	}
	trainer.invFreq = hostmath.AxisRotaryInvFreq(cfg.Policy.RotaryFrequencyBase,
		trainer.axisChannels)
	trainer.positions = make([][3]int, geometry.Seq)
	spatial := geometry.Grid[1] * geometry.Grid[2]
	for token := 0; token < geometry.Seq; token++ {
		trainer.positions[token] = [3]int{
			token / spatial,
			token % spatial / geometry.Grid[2],
			token % geometry.Grid[2],
		}
	}

	specs := ditTensorSpecs(cfg, textDim)
	total := tensor.FirstOffset
	groups := make([]optimizer.GroupSpec, len(specs))
	for i, spec := range specs {
		size, ok := checked.MulInt(spec.rows, spec.cols)
		if !ok {
			return nil, fmt.Errorf("dit train: tensor %s geometry overflows", spec.name)
		}
		end, ok := checked.AddInt(total, size)
		if !ok {
			return nil, fmt.Errorf("dit train: packed parameter geometry overflows")
		}
		groups[i] = optimizer.GroupSpec{
			Name: spec.name, Start: total, End: end,
			Rows: spec.rows, Cols: spec.cols,
		}
		total = end
	}
	compiled, err := optimizer.CompilePlan(total, groups)
	if err != nil {
		return nil, err
	}
	trainer.weights = make([]float32, total)
	trainer.gradients = make([]float32, total)
	for i, spec := range specs {
		values, ok := tensors[spec.name]
		if !ok {
			return nil, fmt.Errorf("dit train: tensor %s is absent", spec.name)
		}
		if err := checked.Length(values, spec.rows, spec.cols); err != nil {
			return nil, fmt.Errorf("dit train: tensor %s: %w", spec.name, err)
		}
		copy(trainer.weights[groups[i].Start:groups[i].End], values)
		trainer.spans[spec.name] = ditSpan{groups[i].Start, groups[i].End}
		// Release the loader storage — the masters own it now.
		delete(tensors, spec.name)
	}
	trainer.optCfg = optimizer.Config{
		BaseLearningRate: trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(total),
		Momentum:         trainingprogram.BuiltinOptimizerPolicy().Momentum(),
		Schedule:         optimizer.ScheduleConstant,
	}
	trainer.stepper, err = optimizer.NewStepper(trainer.weights, trainer.gradients, compiled, trainer.optCfg)
	if err != nil {
		return nil, err
	}
	return trainer, nil
}

// ParameterCount reports the packed trainable parameter total.
func (t *DiTTrainer) ParameterCount() int { return len(t.weights) }

// Config exposes the derived hyperparameters for evidence.
func (t *DiTTrainer) Config() optimizer.Config { return t.optCfg }

// Close releases the stepper backend.
func (t *DiTTrainer) Close() error { return t.stepper.Close() }

func (t *DiTTrainer) view(name string) []float32 {
	s, ok := t.spans[name]
	if !ok {
		panic("dit train: unknown tensor " + name)
	}
	return t.weights[s.start:s.end]
}

func (t *DiTTrainer) gradView(name string) []float32 {
	s, ok := t.spans[name]
	if !ok {
		panic("dit train: unknown tensor " + name)
	}
	return t.gradients[s.start:s.end]
}

// ditTextActs: retained text-projection intermediates. The context is the
// serving contract: the active token rows plus ONE zero-input pad-source row
// broadcast to the fixed text length (the projection of a zero encoder row).
type ditTextActs struct {
	rawFull    []float32 // [activeRows, textDim]; final row zero when padding
	activeRows int
	tokens     int
	hiddenPre  []float32 // [activeRows, d] pre-GELU
	hidden     []float32 // post-GELU
	projected  []float32 // [activeRows, d]
	context    []float32 // [textLen, d]
}

func (t *DiTTrainer) projectText(rawText []float32, tokens int) (*ditTextActs, error) {
	d, textLen := t.cfg.Dim, t.cfg.TextLen
	activeRows, err := representation.ActivePrefixRows(tokens, textLen)
	if err != nil {
		return nil, fmt.Errorf("dit train: raw text: %w", err)
	}
	if err := checked.Length(rawText, tokens, t.textDim); err != nil {
		return nil, fmt.Errorf("dit train: raw text: %w", err)
	}
	a := &ditTextActs{activeRows: activeRows, tokens: tokens}
	a.rawFull = make([]float32, activeRows*t.textDim)
	copy(a.rawFull, rawText)
	a.hiddenPre = make([]float32, activeRows*d)
	hostmath.LinearF64(a.hiddenPre, a.rawFull, t.view("text_embedding.0.weight"), t.view("text_embedding.0.bias"), activeRows, t.textDim, d)
	a.hidden = append([]float32(nil), a.hiddenPre...)
	hostmath.GELUTanhInPlace(a.hidden)
	a.projected = make([]float32, activeRows*d)
	hostmath.LinearF64(a.projected, a.hidden, t.view("text_embedding.2.weight"), t.view("text_embedding.2.bias"), activeRows, d, d)
	a.context = make([]float32, textLen*d)
	copy(a.context, a.projected[:tokens*d])
	if tokens < textLen {
		pad := a.projected[tokens*d : (tokens+1)*d]
		for r := tokens; r < textLen; r++ {
			copy(a.context[r*d:(r+1)*d], pad)
		}
	}
	return a, nil
}

func (t *DiTTrainer) projectTextBackward(a *ditTextActs, dContext []float32) {
	d, textLen := t.cfg.Dim, t.cfg.TextLen
	dProjected := make([]float32, a.activeRows*d)
	copy(dProjected, dContext[:a.tokens*d])
	if a.tokens < textLen {
		pad := dProjected[a.tokens*d : (a.tokens+1)*d]
		for r := a.tokens; r < textLen; r++ {
			row := dContext[r*d : (r+1)*d]
			for i := range pad {
				pad[i] += row[i]
			}
		}
	}
	hostmath.LinearWeightGradient(t.gradView("text_embedding.2.weight"), a.hidden, dProjected, a.activeRows, d, d)
	hostmath.AddBiasGradientF32(t.gradView("text_embedding.2.bias"), dProjected, a.activeRows, d)
	dHidden := make([]float32, a.activeRows*d)
	hostmath.LinearBackwardInput(dHidden, dProjected, t.view("text_embedding.2.weight"), a.activeRows, d, d)
	hostmath.GELUTanhBackward(dHidden, a.hiddenPre, dHidden)
	hostmath.LinearWeightGradient(t.gradView("text_embedding.0.weight"), a.rawFull, dHidden, a.activeRows, t.textDim, d)
	hostmath.AddBiasGradientF32(t.gradView("text_embedding.0.bias"), dHidden, a.activeRows, d)
}

// ditTimeActs: retained timestep-conditioning intermediates. The forward
// mirrors CompileTimestepConditioning stage for stage (same LinearF64/SiLU
// primitives), so serving parity holds bit-for-bit.
type ditTimeActs struct {
	freq      []float32 // [freqDim] sinusoidal embedding (data)
	embed0Pre []float32 // [d] pre-SiLU
	embed0Act []float32
	headE     []float32 // [d]
	headEAct  []float32 // SiLU(headE)
	blockE    []float32 // [6d]
}

func (t *DiTTrainer) timestepConditioning(timestep float64) (*ditTimeActs, error) {
	if !checked.Finite64(timestep) {
		return nil, fmt.Errorf("dit train: timestep is non-finite")
	}
	d, freqDim := t.cfg.Dim, t.cfg.FreqDim
	a := &ditTimeActs{freq: make([]float32, freqDim)}
	half, ok := checked.DivExactInt(freqDim, tensor.PairedExtent)
	if !ok {
		return nil, fmt.Errorf("dit train: timestep embedding width must be even")
	}
	logPeriod := math.Log(float64(t.cfg.Policy.SinusoidalPeriod))
	for i := range half {
		angle := timestep * math.Exp(-logPeriod*float64(i)/float64(half))
		a.freq[i] = float32(math.Cos(angle))
		a.freq[half+i] = float32(math.Sin(angle))
	}
	a.embed0Pre = make([]float32, d)
	hostmath.LinearF64(a.embed0Pre, a.freq, t.view("time_embedding.0.weight"), t.view("time_embedding.0.bias"), tensor.SingletonExtent, freqDim, d)
	a.embed0Act = append([]float32(nil), a.embed0Pre...)
	hostmath.SiLUInPlace(a.embed0Act)
	a.headE = make([]float32, d)
	hostmath.LinearF64(a.headE, a.embed0Act, t.view("time_embedding.2.weight"), t.view("time_embedding.2.bias"), tensor.SingletonExtent, d, d)
	a.headEAct = append([]float32(nil), a.headE...)
	hostmath.SiLUInPlace(a.headEAct)
	blockWidth := media.PairedShiftScaleGateWidth(d)
	a.blockE = make([]float32, blockWidth)
	hostmath.LinearF64(a.blockE, a.headEAct, t.view("time_projection.1.weight"), t.view("time_projection.1.bias"), tensor.SingletonExtent, d, blockWidth)
	return a, nil
}

// timestepBackward: dBlockE accumulates from all blocks' modulation chunks;
// dHeadE from the head's shift and scale paths. Both flow into the shared
// time_projection and time_embedding masters.
func (t *DiTTrainer) timestepBackward(a *ditTimeActs, dBlockE, dHeadE []float32) {
	d := t.cfg.Dim
	blockWidth := media.PairedShiftScaleGateWidth(d)
	hostmath.LinearWeightGradient(t.gradView("time_projection.1.weight"), a.headEAct, dBlockE, tensor.SingletonExtent, d, blockWidth)
	hostmath.AddBiasGradientF32(t.gradView("time_projection.1.bias"), dBlockE, tensor.SingletonExtent, blockWidth)
	dHeadEAct := make([]float32, d)
	hostmath.LinearBackwardInput(dHeadEAct, dBlockE, t.view("time_projection.1.weight"), tensor.SingletonExtent, d, blockWidth)
	dHeadETotal := make([]float32, d)
	hostmath.SiLUBackward(dHeadETotal, a.headE, dHeadEAct)
	for i := range dHeadETotal {
		dHeadETotal[i] += dHeadE[i]
	}
	hostmath.LinearWeightGradient(t.gradView("time_embedding.2.weight"), a.embed0Act, dHeadETotal, tensor.SingletonExtent, d, d)
	hostmath.AddBiasGradientF32(t.gradView("time_embedding.2.bias"), dHeadETotal, tensor.SingletonExtent, d)
	dEmbed0Act := make([]float32, d)
	hostmath.LinearBackwardInput(dEmbed0Act, dHeadETotal, t.view("time_embedding.2.weight"), tensor.SingletonExtent, d, d)
	hostmath.SiLUBackward(dEmbed0Act, a.embed0Pre, dEmbed0Act)
	hostmath.LinearWeightGradient(t.gradView("time_embedding.0.weight"), a.freq, dEmbed0Act, tensor.SingletonExtent, t.cfg.FreqDim, d)
	hostmath.AddBiasGradientF32(t.gradView("time_embedding.0.bias"), dEmbed0Act, tensor.SingletonExtent, d)
}

// ditCrossActs: one block's fixed-context cross K/V (trainable projections).
type ditCrossActs struct {
	kProj []float32 // [textLen, d] pre-norm
	k     []float32 // post norm_k
	v     []float32
}

func (t *DiTTrainer) crossContext(prefix string, context []float32) ditCrossActs {
	d, textLen := t.cfg.Dim, t.cfg.TextLen
	a := ditCrossActs{
		kProj: make([]float32, textLen*d),
		k:     make([]float32, textLen*d),
		v:     make([]float32, textLen*d),
	}
	hostmath.LinearF64(a.kProj, context, t.view(prefix+"k.weight"), t.view(prefix+"k.bias"), textLen, d, d)
	hostmath.RMSNormInto(a.k, a.kProj, t.view(prefix+"norm_k.weight"), textLen, d, t.cfg.Eps)
	hostmath.LinearF64(a.v, context, t.view(prefix+"v.weight"), t.view(prefix+"v.bias"), textLen, d, d)
	return a
}

// ditBlockActs: one block's retained training-precision intermediates.
type ditBlockActs struct {
	input    []float32 // [seq, d] block input (previous output)
	lnSelf   []float32 // LN(input), pre-modulation
	selfIn   []float32 // modulated
	qProj    []float32 // pre full-width QK norm
	kProj    []float32
	vProj    []float32
	qNorm    []float32 // post norm, pre rope
	kNorm    []float32
	qRope    []float32
	kRope    []float32
	attnOut  []float32 // SDPA output [seq, d]
	selfPro  []float32 // o-projection
	selfRes  []float32
	crossIn  []float32 // affine LN(selfRes)
	cQProj   []float32 // cross query pre-norm
	cQ       []float32 // post norm_q
	cAttn    []float32 // cross SDPA output
	cPro     []float32 // cross o-projection
	crossRes []float32
	lnFFN    []float32
	ffnIn    []float32
	ffnPre   []float32 // [seq, ffn]
	ffnAct   []float32
	ffnOut   []float32
	output   []float32
	cross    ditCrossActs
	mVec     [6][]float32 // modulation + blockE chunk vectors
}

func (t *DiTTrainer) attentionScale() float64 {
	return 1 / math.Sqrt(float64(t.cfg.Dim/t.cfg.NumHeads))
}

func (t *DiTTrainer) blockForward(layer int, input, blockE []float32, cross ditCrossActs) ditBlockActs {
	cfg := t.cfg
	d, f := cfg.Dim, cfg.FFNDim
	seq, textLen := t.geometry.Seq, cfg.TextLen
	heads := cfg.NumHeads
	hd := d / heads
	eps := cfg.Eps
	prefix := denoiserBlockPrefix(layer)
	modulation := t.view(prefix + "modulation")
	a := ditBlockActs{input: input, cross: cross}
	for chunk := range media.DefaultPairedShiftScaleGateFields() {
		vec := make([]float32, d)
		for i := range vec {
			vec[i] = modulation[chunk*d+i] + blockE[chunk*d+i]
		}
		a.mVec[chunk] = vec
	}

	a.lnSelf = make([]float32, seq*d)
	hostmath.LayerNormInto(a.lnSelf, input, nil, nil, seq, d, eps)
	a.selfIn = make([]float32, seq*d)
	offsets := media.PairedShiftFirstGateOffsets()
	hostmath.AdaptiveShiftScale(a.selfIn, a.lnSelf, a.mVec[offsets.PreShift], a.mVec[offsets.PreScale], seq, d)

	self := prefix + "self_attn."
	a.qProj = make([]float32, seq*d)
	hostmath.LinearF64(a.qProj, a.selfIn, t.view(self+"q.weight"), t.view(self+"q.bias"), seq, d, d)
	a.kProj = make([]float32, seq*d)
	hostmath.LinearF64(a.kProj, a.selfIn, t.view(self+"k.weight"), t.view(self+"k.bias"), seq, d, d)
	a.vProj = make([]float32, seq*d)
	hostmath.LinearF64(a.vProj, a.selfIn, t.view(self+"v.weight"), t.view(self+"v.bias"), seq, d, d)
	a.qNorm = make([]float32, seq*d)
	hostmath.RMSNormInto(a.qNorm, a.qProj, t.view(self+"norm_q.weight"), seq, d, eps)
	a.kNorm = make([]float32, seq*d)
	hostmath.RMSNormInto(a.kNorm, a.kProj, t.view(self+"norm_k.weight"), seq, d, eps)
	a.qRope = append([]float32(nil), a.qNorm...)
	a.kRope = append([]float32(nil), a.kNorm...)
	for token := 0; token < seq; token++ {
		for head := 0; head < heads; head++ {
			offset := token*d + head*hd
			hostmath.ApplyAxisRotaryInterleaved(a.qRope[offset:offset+hd], t.axisChannels, t.invFreq, t.positions[token])
			hostmath.ApplyAxisRotaryInterleaved(a.kRope[offset:offset+hd], t.axisChannels, t.invFreq, t.positions[token])
		}
	}
	a.attnOut = make([]float32, seq*d)
	hostmath.ScaledMaskedBidirectionalAttention(a.attnOut, a.qRope, a.kRope, a.vProj, seq, seq, heads, heads, hd, t.attentionScale(), nil)
	a.selfPro = make([]float32, seq*d)
	hostmath.LinearF64(a.selfPro, a.attnOut, t.view(self+"o.weight"), t.view(self+"o.bias"), seq, d, d)
	a.selfRes = make([]float32, seq*d)
	gate2 := a.mVec[offsets.PreGate]
	for r := 0; r < seq; r++ {
		for i := 0; i < d; i++ {
			a.selfRes[r*d+i] = input[r*d+i] + a.selfPro[r*d+i]*gate2[i]
		}
	}

	crossPrefix := prefix + "cross_attn."
	a.crossIn = make([]float32, seq*d)
	hostmath.LayerNormInto(a.crossIn, a.selfRes, t.view(prefix+"norm3.weight"), t.view(prefix+"norm3.bias"), seq, d, eps)
	a.cQProj = make([]float32, seq*d)
	hostmath.LinearF64(a.cQProj, a.crossIn, t.view(crossPrefix+"q.weight"), t.view(crossPrefix+"q.bias"), seq, d, d)
	a.cQ = make([]float32, seq*d)
	hostmath.RMSNormInto(a.cQ, a.cQProj, t.view(crossPrefix+"norm_q.weight"), seq, d, eps)
	a.cAttn = make([]float32, seq*d)
	hostmath.ScaledMaskedBidirectionalAttention(a.cAttn, a.cQ, cross.k, cross.v, seq, textLen, heads, heads, hd, t.attentionScale(), nil)
	a.cPro = make([]float32, seq*d)
	hostmath.LinearF64(a.cPro, a.cAttn, t.view(crossPrefix+"o.weight"), t.view(crossPrefix+"o.bias"), seq, d, d)
	a.crossRes = make([]float32, seq*d)
	for i := range a.crossRes {
		a.crossRes[i] = a.selfRes[i] + a.cPro[i]
	}

	a.lnFFN = make([]float32, seq*d)
	hostmath.LayerNormInto(a.lnFFN, a.crossRes, nil, nil, seq, d, eps)
	a.ffnIn = make([]float32, seq*d)
	hostmath.AdaptiveShiftScale(a.ffnIn, a.lnFFN, a.mVec[offsets.PostShift], a.mVec[offsets.PostScale], seq, d)
	a.ffnPre = make([]float32, seq*f)
	hostmath.LinearF64(a.ffnPre, a.ffnIn, t.view(prefix+"ffn.0.weight"), t.view(prefix+"ffn.0.bias"), seq, d, f)
	a.ffnAct = append([]float32(nil), a.ffnPre...)
	hostmath.GELUTanhInPlace(a.ffnAct)
	a.ffnOut = make([]float32, seq*d)
	hostmath.LinearF64(a.ffnOut, a.ffnAct, t.view(prefix+"ffn.2.weight"), t.view(prefix+"ffn.2.bias"), seq, f, d)
	a.output = make([]float32, seq*d)
	gate5 := a.mVec[offsets.PostGate]
	for r := 0; r < seq; r++ {
		for i := 0; i < d; i++ {
			a.output[r*d+i] = a.crossRes[r*d+i] + a.ffnOut[r*d+i]*gate5[i]
		}
	}
	return a
}

// blockBackward: full VJP of blockForward. Weight gradients accumulate into
// the packed masters; the modulation chunk gradients accumulate into BOTH
// the block's modulation grad and the shared dBlockE; cross K/V gradients
// accumulate into the shared dContext. Returns the gradient at the block
// input.
func (t *DiTTrainer) blockBackward(layer int, a *ditBlockActs, dOutput, dBlockE, dContext []float32) []float32 {
	cfg := t.cfg
	d, f := cfg.Dim, cfg.FFNDim
	seq, textLen := t.geometry.Seq, cfg.TextLen
	heads := cfg.NumHeads
	hd := d / heads
	eps := cfg.Eps
	prefix := denoiserBlockPrefix(layer)
	gradModulation := t.gradView(prefix + "modulation")
	offsets := media.PairedShiftFirstGateOffsets()
	chunkGrad := func(chunk int, add []float32) {
		for i := 0; i < d; i++ {
			gradModulation[chunk*d+i] += add[i]
			dBlockE[chunk*d+i] += add[i]
		}
	}

	// output = crossRes + ffnOut*gate5.
	dCrossRes := append([]float32(nil), dOutput...)
	dFFNOut := make([]float32, seq*d)
	dGate := make([]float32, d)
	gate5 := a.mVec[offsets.PostGate]
	for r := 0; r < seq; r++ {
		for i := 0; i < d; i++ {
			g := dOutput[r*d+i]
			dFFNOut[r*d+i] = g * gate5[i]
			dGate[i] += g * a.ffnOut[r*d+i]
		}
	}
	chunkGrad(offsets.PostGate, dGate)

	// Feed-forward.
	hostmath.LinearWeightGradient(t.gradView(prefix+"ffn.2.weight"), a.ffnAct, dFFNOut, seq, f, d)
	hostmath.AddBiasGradientF32(t.gradView(prefix+"ffn.2.bias"), dFFNOut, seq, d)
	dFFNAct := make([]float32, seq*f)
	hostmath.LinearBackwardInput(dFFNAct, dFFNOut, t.view(prefix+"ffn.2.weight"), seq, f, d)
	hostmath.GELUTanhBackward(dFFNAct, a.ffnPre, dFFNAct)
	hostmath.LinearWeightGradient(t.gradView(prefix+"ffn.0.weight"), a.ffnIn, dFFNAct, seq, d, f)
	hostmath.AddBiasGradientF32(t.gradView(prefix+"ffn.0.bias"), dFFNAct, seq, f)
	dFFNIn := make([]float32, seq*d)
	hostmath.LinearBackwardInput(dFFNIn, dFFNAct, t.view(prefix+"ffn.0.weight"), seq, d, f)

	dShift := make([]float32, d)
	dScale := make([]float32, d)
	dLN := make([]float32, seq*d)
	hostmath.AdaptiveShiftScaleBackward(dLN, dShift, dScale, a.lnFFN, a.mVec[offsets.PostScale], dFFNIn, seq, d)
	chunkGrad(offsets.PostShift, dShift)
	chunkGrad(offsets.PostScale, dScale)
	hostmath.LayerNormBackward(dCrossRes, nil, nil, a.crossRes, nil, dLN, seq, d, eps, true)

	// crossRes = selfRes + crossProjected.
	dSelfRes := append([]float32(nil), dCrossRes...)
	crossPrefix := prefix + "cross_attn."
	hostmath.LinearWeightGradient(t.gradView(crossPrefix+"o.weight"), a.cAttn, dCrossRes, seq, d, d)
	hostmath.AddBiasGradientF32(t.gradView(crossPrefix+"o.bias"), dCrossRes, seq, d)
	dCAttn := make([]float32, seq*d)
	hostmath.LinearBackwardInput(dCAttn, dCrossRes, t.view(crossPrefix+"o.weight"), seq, d, d)

	dCQ := make([]float32, seq*d)
	dK := make([]float32, textLen*d)
	dV := make([]float32, textLen*d)
	hostmath.ScaledMaskedBidirectionalAttentionBackward(dCQ, dK, dV, a.cQ, a.cross.k, a.cross.v, dCAttn, seq, textLen, heads, heads, hd, t.attentionScale(), nil)

	// Cross K/V into the shared context gradient.
	dKProj := make([]float32, textLen*d)
	hostmath.RMSNormBackward(dKProj, t.gradView(crossPrefix+"norm_k.weight"), a.cross.kProj, t.view(crossPrefix+"norm_k.weight"), dK, textLen, d, eps, false)
	hostmath.LinearWeightGradient(t.gradView(crossPrefix+"k.weight"), t.currentContext, dKProj, textLen, d, d)
	hostmath.AddBiasGradientF32(t.gradView(crossPrefix+"k.bias"), dKProj, textLen, d)
	dCtx := make([]float32, textLen*d)
	hostmath.LinearBackwardInput(dCtx, dKProj, t.view(crossPrefix+"k.weight"), textLen, d, d)
	for i := range dContext {
		dContext[i] += dCtx[i]
	}
	hostmath.LinearWeightGradient(t.gradView(crossPrefix+"v.weight"), t.currentContext, dV, textLen, d, d)
	hostmath.AddBiasGradientF32(t.gradView(crossPrefix+"v.bias"), dV, textLen, d)
	hostmath.LinearBackwardInput(dCtx, dV, t.view(crossPrefix+"v.weight"), textLen, d, d)
	for i := range dContext {
		dContext[i] += dCtx[i]
	}

	// Cross query and the affine pre-cross norm.
	dCQProj := make([]float32, seq*d)
	hostmath.RMSNormBackward(dCQProj, t.gradView(crossPrefix+"norm_q.weight"), a.cQProj, t.view(crossPrefix+"norm_q.weight"), dCQ, seq, d, eps, false)
	hostmath.LinearWeightGradient(t.gradView(crossPrefix+"q.weight"), a.crossIn, dCQProj, seq, d, d)
	hostmath.AddBiasGradientF32(t.gradView(crossPrefix+"q.bias"), dCQProj, seq, d)
	dCrossIn := make([]float32, seq*d)
	hostmath.LinearBackwardInput(dCrossIn, dCQProj, t.view(crossPrefix+"q.weight"), seq, d, d)
	hostmath.LayerNormBackward(dSelfRes, t.gradView(prefix+"norm3.weight"), t.gradView(prefix+"norm3.bias"),
		a.selfRes, t.view(prefix+"norm3.weight"), dCrossIn, seq, d, eps, true)

	// selfRes = input + selfProjected*gate2.
	dInput := append([]float32(nil), dSelfRes...)
	dSelfPro := make([]float32, seq*d)
	clear(dGate)
	gate2 := a.mVec[offsets.PreGate]
	for r := 0; r < seq; r++ {
		for i := 0; i < d; i++ {
			g := dSelfRes[r*d+i]
			dSelfPro[r*d+i] = g * gate2[i]
			dGate[i] += g * a.selfPro[r*d+i]
		}
	}
	chunkGrad(offsets.PreGate, dGate)

	self := prefix + "self_attn."
	hostmath.LinearWeightGradient(t.gradView(self+"o.weight"), a.attnOut, dSelfPro, seq, d, d)
	hostmath.AddBiasGradientF32(t.gradView(self+"o.bias"), dSelfPro, seq, d)
	dAttnOut := make([]float32, seq*d)
	hostmath.LinearBackwardInput(dAttnOut, dSelfPro, t.view(self+"o.weight"), seq, d, d)

	dQRope := make([]float32, seq*d)
	dKRope := make([]float32, seq*d)
	dVProj := make([]float32, seq*d)
	hostmath.ScaledMaskedBidirectionalAttentionBackward(dQRope, dKRope, dVProj, a.qRope, a.kRope, a.vProj, dAttnOut, seq, seq, heads, heads, hd, t.attentionScale(), nil)
	for token := 0; token < seq; token++ {
		for head := 0; head < heads; head++ {
			offset := token*d + head*hd
			hostmath.AxisRotaryInterleavedBackward(dQRope[offset:offset+hd], t.axisChannels, t.invFreq, t.positions[token])
			hostmath.AxisRotaryInterleavedBackward(dKRope[offset:offset+hd], t.axisChannels, t.invFreq, t.positions[token])
		}
	}
	dQProj := make([]float32, seq*d)
	hostmath.RMSNormBackward(dQProj, t.gradView(self+"norm_q.weight"), a.qProj, t.view(self+"norm_q.weight"), dQRope, seq, d, eps, false)
	dKProjSelf := make([]float32, seq*d)
	hostmath.RMSNormBackward(dKProjSelf, t.gradView(self+"norm_k.weight"), a.kProj, t.view(self+"norm_k.weight"), dKRope, seq, d, eps, false)

	dSelfIn := make([]float32, seq*d)
	scratch := make([]float32, seq*d)
	for _, source := range []struct {
		grad []float32
		name string
	}{
		{dQProj, "q"}, {dKProjSelf, "k"}, {dVProj, "v"},
	} {
		hostmath.LinearWeightGradient(t.gradView(self+source.name+".weight"), a.selfIn, source.grad, seq, d, d)
		hostmath.AddBiasGradientF32(t.gradView(self+source.name+".bias"), source.grad, seq, d)
		hostmath.LinearBackwardInput(scratch, source.grad, t.view(self+source.name+".weight"), seq, d, d)
		for i := range dSelfIn {
			dSelfIn[i] += scratch[i]
		}
	}

	clear(dShift)
	clear(dScale)
	hostmath.AdaptiveShiftScaleBackward(dLN, dShift, dScale, a.lnSelf, a.mVec[offsets.PreScale], dSelfIn, seq, d)
	chunkGrad(offsets.PreShift, dShift)
	chunkGrad(offsets.PreScale, dScale)
	hostmath.LayerNormBackward(dInput, nil, nil, a.input, nil, dLN, seq, d, eps, true)
	return dInput
}

// ditForward: one full retained forward pass.
type ditForward struct {
	patchTokens []float32
	text        *ditTextActs
	time        *ditTimeActs
	blocks      []ditBlockActs
	hidden0     []float32 // patch embedding output
	lnHead      []float32
	headShift   []float32
	headScale   []float32
	modulated   []float32
	patches     []float32 // [seq, patchOut]
	vLatent     []float32 // [OutDim, F, H, W] predicted velocity
}

// currentContext: the context rows the running backward consumes (set by the
// forward that produced the activations).
// Kept on the trainer to avoid threading it through every block call.

func (t *DiTTrainer) patchify(latent []float32) ([]float32, error) {
	g, c := t.geometry, t.cfg
	if len(latent) != g.Elements() {
		return nil, fmt.Errorf("dit train: latent has %d elements, need %d", len(latent), g.Elements())
	}
	out := make([]float32, g.Seq*c.patchIn())
	hostmath.PatchifyChannelMajor(out, latent, c.InDim, g.LatentFrames, g.LatentHeight, g.LatentWidth,
		c.PatchSize[0], c.PatchSize[1], c.PatchSize[2])
	return out, nil
}

// forwardConditioned: patch embedding -> every block -> modulated head ->
// unpatchified velocity, from an already-projected context and computed
// conditioning (the seam the parity gate drives with committed goldens).
func (t *DiTTrainer) forwardConditioned(latent, context, blockE, headE []float32) (*ditForward, error) {
	cfg, g := t.cfg, t.geometry
	d := cfg.Dim
	seq := g.Seq
	if len(context) != cfg.TextLen*d || len(blockE) != 6*d || len(headE) != d {
		return nil, fmt.Errorf("dit train: conditioning shapes context=%d blockE=%d headE=%d", len(context), len(blockE), len(headE))
	}
	patchTokens, err := t.patchify(latent)
	if err != nil {
		return nil, err
	}
	state := &ditForward{patchTokens: patchTokens}
	t.currentContext = context
	state.hidden0 = make([]float32, seq*d)
	hostmath.LinearF64(state.hidden0, patchTokens, t.view("patch_embedding.weight"), t.view("patch_embedding.bias"), seq, cfg.patchIn(), d)
	hidden := state.hidden0
	state.blocks = make([]ditBlockActs, cfg.NumLayers)
	for layer := 0; layer < cfg.NumLayers; layer++ {
		cross := t.crossContext(denoiserBlockPrefix(layer)+"cross_attn.", context)
		state.blocks[layer] = t.blockForward(layer, hidden, blockE, cross)
		hidden = state.blocks[layer].output
	}
	state.lnHead = make([]float32, seq*d)
	hostmath.LayerNormInto(state.lnHead, hidden, nil, nil, seq, d, cfg.Eps)
	headModulation := t.view("head.modulation")
	state.headShift = make([]float32, d)
	state.headScale = make([]float32, d)
	for i := range d {
		state.headShift[i] = headModulation[i] + headE[i]
		state.headScale[i] = headModulation[d+i] + headE[i]
	}
	state.modulated = make([]float32, seq*d)
	hostmath.AdaptiveShiftScale(state.modulated, state.lnHead, state.headShift, state.headScale, seq, d)
	state.patches = make([]float32, seq*cfg.patchOut())
	hostmath.LinearF64(state.patches, state.modulated, t.view("head.head.weight"), t.view("head.head.bias"), seq, d, cfg.patchOut())
	outputFrames := g.Grid[0] * cfg.PatchSize[0]
	state.vLatent = make([]float32, cfg.OutDim*outputFrames*g.LatentHeight*g.LatentWidth)
	hostmath.UnpatchifyChannelMajor(state.vLatent, state.patches, cfg.OutDim, outputFrames, g.LatentHeight, g.LatentWidth,
		cfg.PatchSize[0], cfg.PatchSize[1], cfg.PatchSize[2])
	return state, nil
}

func (t *DiTTrainer) forward(batch DiTTrainBatch) (*ditForward, error) {
	text, err := t.projectText(batch.RawText, batch.TextTokens)
	if err != nil {
		return nil, err
	}
	timeActs, err := t.timestepConditioning(batch.Timestep)
	if err != nil {
		return nil, err
	}
	state, err := t.forwardConditioned(batch.Latent, text.context, timeActs.blockE, timeActs.headE)
	if err != nil {
		return nil, err
	}
	state.text, state.time = text, timeActs
	return state, nil
}

// Loss: the training-precision forward and flow-matching MSE without
// touching gradients.
func (t *DiTTrainer) Loss(batch DiTTrainBatch) (float64, error) {
	state, err := t.forward(batch)
	if err != nil {
		return 0, err
	}
	loss, _, err := trainingprogram.MeanSquaredErrorF32(state.vLatent, batch.Target, false)
	return loss, err
}

// Step: one observed full-DiT training step — forward, flow-matching MSE,
// full VJP into every packed master, Muon update.
func (t *DiTTrainer) Step(batch DiTTrainBatch) (DiTTrainStepResult, error) {
	loss, gradientL2, err := t.lossAndGradients(batch)
	if err != nil {
		return DiTTrainStepResult{}, err
	}
	return optimizer.Advance(t.stepper, t.optCfg, &t.step, loss, gradientL2)
}

func (t *DiTTrainer) lossAndGradients(batch DiTTrainBatch) (float64, float64, error) {
	state, err := t.forward(batch)
	if err != nil {
		return 0, 0, err
	}
	loss, dV, err := trainingprogram.MeanSquaredErrorF32(state.vLatent, batch.Target, true)
	if err != nil {
		return 0, 0, err
	}
	clear(t.gradients)
	cfg, g := t.cfg, t.geometry
	d, seq := cfg.Dim, g.Seq

	// Head backward.
	outputFrames := g.Grid[0] * cfg.PatchSize[0]
	dPatches := make([]float32, seq*cfg.patchOut())
	hostmath.UnpatchifyChannelMajorTranspose(dPatches, dV, cfg.OutDim, outputFrames, g.LatentHeight, g.LatentWidth,
		cfg.PatchSize[0], cfg.PatchSize[1], cfg.PatchSize[2])
	hostmath.LinearWeightGradient(t.gradView("head.head.weight"), state.modulated, dPatches, seq, d, cfg.patchOut())
	hostmath.AddBiasGradientF32(t.gradView("head.head.bias"), dPatches, seq, cfg.patchOut())
	dModulated := make([]float32, seq*d)
	hostmath.LinearBackwardInput(dModulated, dPatches, t.view("head.head.weight"), seq, d, cfg.patchOut())
	dShift := make([]float32, d)
	dScale := make([]float32, d)
	dLNHead := make([]float32, seq*d)
	hostmath.AdaptiveShiftScaleBackward(dLNHead, dShift, dScale, state.lnHead, state.headScale, dModulated, seq, d)
	gradHeadModulation := t.gradView("head.modulation")
	dHeadE := make([]float32, d)
	for i := 0; i < d; i++ {
		gradHeadModulation[i] += dShift[i]
		gradHeadModulation[d+i] += dScale[i]
		dHeadE[i] = dShift[i] + dScale[i]
	}
	lastBlock, ok := checked.Last(state.blocks)
	if !ok {
		return 0, 0, fmt.Errorf("dit train: transformer block activations are absent")
	}
	lastOutput := lastBlock.output
	dHidden := make([]float32, seq*d)
	hostmath.LayerNormBackward(dHidden, nil, nil, lastOutput, nil, dLNHead, seq, d, cfg.Eps, false)

	// Blocks in reverse; shared conditioning/context gradients accumulate.
	dBlockE := make([]float32, media.PairedShiftScaleGateWidth(d))
	dContext := make([]float32, cfg.TextLen*d)
	for reverse := range cfg.NumLayers {
		layer := checked.ReverseIndex(reverse, cfg.NumLayers)
		dHidden = t.blockBackward(layer, &state.blocks[layer], dHidden, dBlockE, dContext)
	}

	// Patch embedding (the latent input is data; no input gradient needed).
	hostmath.LinearWeightGradient(t.gradView("patch_embedding.weight"), state.patchTokens, dHidden, seq, cfg.patchIn(), d)
	hostmath.AddBiasGradientF32(t.gradView("patch_embedding.bias"), dHidden, seq, d)

	// Shared conditioning paths.
	t.timestepBackward(state.time, dBlockE, dHeadE)
	t.projectTextBackward(state.text, dContext)

	var gradientSquared float64
	for _, g := range t.gradients {
		gradientSquared += float64(g) * float64(g)
	}
	return loss, math.Sqrt(gradientSquared), nil
}

// ProjectTextContext: the trainer's projected [textLen, dim] context for one
// raw text row set — the evidence surface probes log against the committed
// projected-context capture (Loss/Step run the same path internally).
func (t *DiTTrainer) ProjectTextContext(rawText []float32, tokens int) ([]float32, error) {
	acts, err := t.projectText(rawText, tokens)
	if err != nil {
		return nil, err
	}
	return acts.context, nil
}

// LoadDiTTrainerTensors: the complete trainable tensor set from an F32
// safetensors checkpoint directory (all denoiser tensors plus the text
// projection), plus the raw text width. The returned map is consumed by
// NewDiTTrainer.
func LoadDiTTrainerTensors(dir string, c DenoiserConfig) (map[string][]float32, int, error) {
	weights, err := LoadDenoiserWeights(dir, c)
	if err != nil {
		return nil, 0, err
	}
	projection, _, err := loadProjectionWeights(dir)
	if err != nil {
		return nil, 0, err
	}
	tensors := weights.values
	weights.values = nil
	tensors["text_embedding.0.weight"] = projection.Linear0W
	tensors["text_embedding.0.bias"] = projection.Linear0B
	tensors["text_embedding.2.weight"] = projection.Linear2W
	tensors["text_embedding.2.bias"] = projection.Linear2B
	textDimension, err := checked.Rows(projection.Linear0W, c.Dim)
	if err != nil {
		return nil, 0, fmt.Errorf("dit train: text projection: %w", err)
	}
	return tensors, textDimension, nil
}

// LoadTrainerTensors: the LiveEdit-checkpoint variant — every denoiser
// tensor plus the text projection from the compiled .pt bindings.
func (p ReferenceEditCheckpoint) LoadTrainerTensors() (map[string][]float32, int, error) {
	weights, err := p.LoadWeights()
	if err != nil {
		return nil, 0, err
	}
	projection, _, err := p.loadTextProjection()
	if err != nil {
		return nil, 0, err
	}
	tensors := weights.values
	weights.values = nil
	tensors["text_embedding.0.weight"] = projection.Linear0W
	tensors["text_embedding.0.bias"] = projection.Linear0B
	tensors["text_embedding.2.weight"] = projection.Linear2W
	tensors["text_embedding.2.bias"] = projection.Linear2B
	textDimension, err := checked.Rows(projection.Linear0W, p.Config.Dim)
	if err != nil {
		return nil, 0, fmt.Errorf("dit train: text projection: %w", err)
	}
	return tensors, textDimension, nil
}

// RawTextRows: real frozen-encoder text rows for one prompt — tokenizer,
// streamed relative-position encoder, compacted to the active token rows.
// These are the pre-projection rows the trainable text embedding consumes.
func RawTextRows(spec TextConditioningSpec, prompt string) ([]float32, int, int, error) {
	tok, err := loadFixedUnigramTokenizer(spec.TokenizerDir, spec.SequenceLength)
	if err != nil {
		return nil, 0, 0, err
	}
	catalog, err := pytorchzip.ReadCatalog(spec.EncoderCheckpoint)
	if err != nil {
		return nil, 0, 0, err
	}
	plan, err := CompileEncoderPlan(catalog.Tensors, EncoderConfig{RelativeMaxDistance: spec.RelativeMaxDistance, NormEps: spec.NormEps})
	if err != nil {
		return nil, 0, 0, err
	}
	if !plan.OK {
		return nil, 0, 0, fmt.Errorf("dit train encoder contract: missing=%v unexpected=%v", plan.Missing, plan.Unexpected)
	}
	ids, mask, err := tok.EncodeWithMask(prompt)
	if err != nil {
		return nil, 0, 0, err
	}
	tokens := tensor.FirstOffset
	for _, v := range mask {
		if checked.Nonzero(v) {
			tokens++
		}
	}
	if !checked.PositiveInts(tokens) {
		return nil, 0, 0, fmt.Errorf("dit train: prompt produced no active tokens")
	}
	encoded, _, err := EncodeTokensStreamed(spec.EncoderCheckpoint, plan, ids, mask)
	if err != nil {
		return nil, 0, 0, err
	}
	textDim := plan.Config.Dim
	return encoded[:tokens*textDim], tokens, textDim, nil
}
