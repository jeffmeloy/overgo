package seq2seq

import (
	"errors"
	"fmt"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// TrainingPair is one encoder input and shifted decoder target.
type TrainingPair struct {
	Source       []int
	DecoderInput []int
	Targets      []int
}

// TrainingPair tokenizes one typed paired-text example with model-owned IDs.
func (g *Generator) TrainingPair(example trainingdata.Example) (TrainingPair, error) {
	if g == nil || g.model == nil || g.tokenizer == nil {
		return TrainingPair{}, errors.New("seq2seq: generator is unavailable")
	}
	input, target, err := trainingdata.TextPair(example)
	if err != nil {
		return TrainingPair{}, err
	}
	source, err := g.tokenizer.Encode(input)
	if err != nil {
		return TrainingPair{}, fmt.Errorf("seq2seq: tokenize training input: %w", err)
	}
	answer, err := g.tokenizer.Encode(target)
	if err != nil {
		return TrainingPair{}, fmt.Errorf("seq2seq: tokenize training target: %w", err)
	}
	if len(source) == 0 || len(answer) == 0 {
		return TrainingPair{}, errors.New("seq2seq: training pair tokenized empty")
	}
	decoder := make([]int, len(answer)+1)
	labels := make([]int, len(answer)+1)
	decoder[0] = g.model.Dims.StartToken
	copy(decoder[1:], answer)
	copy(labels, answer)
	labels[len(answer)] = g.model.Dims.EOSToken
	return TrainingPair{Source: source, DecoderInput: decoder, Targets: labels}, nil
}

// Loss evaluates one teacher-forced pair through shared cross-entropy.
func (m *Model) Loss(pair TrainingPair) (float64, error) {
	if m == nil || len(pair.Source) == 0 || len(pair.DecoderInput) == 0 || len(pair.DecoderInput) != len(pair.Targets) {
		return 0, errors.New("seq2seq: invalid training pair")
	}
	for _, target := range pair.Targets {
		if target < 0 || target >= m.Dims.Vocab {
			return 0, fmt.Errorf("seq2seq: training target %d outside vocab %d", target, m.Dims.Vocab)
		}
	}
	memory, err := m.Encode(pair.Source)
	if err != nil {
		return 0, err
	}
	logits, err := m.DecodeFull(memory, len(pair.Source), pair.DecoderInput)
	if err != nil {
		return 0, err
	}
	gradients := make([]float32, len(logits))
	return hostmath.SoftmaxCrossEntropy(gradients, logits, pair.Targets, len(pair.Targets), m.Dims.Vocab), nil
}

const (
	trainingForward  = "seq2seq-forward"
	trainingBackward = "decoder-boundary-backward"
	trainingMuon     = "muon"
)

// Trainer expands Needle training inward from the decoder output boundary.
type Trainer struct {
	model     *Model
	layout    trainingLayout
	bindings  []trainingParameterBinding
	program   trainingprogram.TrainingProgram
	execution trainingprogram.Execution[trainingStep]
	weights   []float32
	gradients []float32
	optimizer trainingOptimizer
}

type trainingOptimizer interface {
	Step() error
	Close() error
}

type trainingStep struct {
	pair                    TrainingPair
	hidden, normed, logits  []float32
	trace                   decoderTrainingTrace
	gradient                []float32
	projectionGradient      []float32
	qProjectionGradient     []float32
	kProjectionGradient     []float32
	vProjectionGradient     []float32
	attentionQGradient      []float32
	attentionKGradient      []float32
	attentionVGradient      []float32
	selfAttentionQGradient  []float32
	selfAttentionKGradient  []float32
	selfAttentionVGradient  []float32
	selfOutputGradient      []float32
	selfQGradient           []float32
	selfKGradient           []float32
	selfVGradient           []float32
	finalSelfOutputGradient []float32
	priorDecoderGradient    []float32
	selfRawGateGradient     float32
	loss                    float64
}

type parameterSpan struct{ start, end int }

type trainingLayout struct {
	finalNorm, rawGate, output            parameterSpan
	inputNorm, qNorm, kNorm               parameterSpan
	qProjection, kProjection, vProjection parameterSpan
	selfRawGate, selfOutput               parameterSpan
	selfInputNorm, selfQNorm, selfKNorm   parameterSpan
	selfQ, selfK, selfV                   parameterSpan
	count                                 int
}

type trainingParameterBinding struct {
	name           string
	span           parameterSpan
	rows, cols     int
	f32            []float32
	bf16           []uint16
	scalar, folded *float32
}

func (b trainingParameterBinding) load(dst []float32) {
	values := dst[b.span.start:b.span.end]
	switch {
	case b.scalar != nil:
		values[0] = *b.scalar
	case b.f32 != nil:
		copy(values, b.f32)
	default:
		copyBF16ToFloat32(values, b.bf16)
	}
}

func (b trainingParameterBinding) publish(src []float32) {
	values := src[b.span.start:b.span.end]
	switch {
	case b.scalar != nil:
		*b.scalar = values[0]
		if b.folded != nil {
			*b.folded = sigmoid(values[0])
		}
	case b.f32 != nil:
		copy(b.f32, values)
	default:
		copyFloat32ToBF16(b.bf16, values)
	}
}

func trainingParameterBindings(model *Model, layout trainingLayout) []trainingParameterBinding {
	cross := &model.decoderCross[len(model.decoderCross)-1]
	self := &model.decoderSelf[len(model.decoderSelf)-1]
	d := model.Dims.DModel
	qWidth := model.Dims.Heads * model.Dims.HeadDim
	kvWidth := model.Dims.KVHeads * model.Dims.HeadDim
	return []trainingParameterBinding{
		{name: "decoder.final_norm", span: layout.finalNorm, rows: d, cols: 1, f32: model.decFinalNorm},
		{name: "decoder.final_cross.raw_gate", span: layout.rawGate, rows: 1, cols: 1, scalar: &cross.rawGate, folded: &cross.gate},
		{name: "decoder.final_cross.output", span: layout.output, rows: d, cols: qWidth, bf16: cross.o},
		{name: "decoder.final_cross.input_norm", span: layout.inputNorm, rows: d, cols: 1, f32: cross.inNorm},
		{name: "decoder.final_cross.q_norm", span: layout.qNorm, rows: model.Dims.HeadDim, cols: 1, f32: cross.qNorm},
		{name: "decoder.final_cross.k_norm", span: layout.kNorm, rows: model.Dims.HeadDim, cols: 1, f32: cross.kNorm},
		{name: "decoder.final_cross.q", span: layout.qProjection, rows: qWidth, cols: d, bf16: cross.q},
		{name: "decoder.final_cross.k", span: layout.kProjection, rows: kvWidth, cols: d, bf16: cross.k},
		{name: "decoder.final_cross.v", span: layout.vProjection, rows: kvWidth, cols: d, bf16: cross.v},
		{name: "decoder.final_self.raw_gate", span: layout.selfRawGate, rows: 1, cols: 1, scalar: &self.rawGate, folded: &self.gate},
		{name: "decoder.final_self.output", span: layout.selfOutput, rows: d, cols: qWidth, bf16: self.o},
		{name: "decoder.final_self.input_norm", span: layout.selfInputNorm, rows: d, cols: 1, f32: self.inNorm},
		{name: "decoder.final_self.q_norm", span: layout.selfQNorm, rows: model.Dims.HeadDim, cols: 1, f32: self.qNorm},
		{name: "decoder.final_self.k_norm", span: layout.selfKNorm, rows: model.Dims.HeadDim, cols: 1, f32: self.kNorm},
		{name: "decoder.final_self.q", span: layout.selfQ, rows: qWidth, cols: d, bf16: self.q},
		{name: "decoder.final_self.k", span: layout.selfK, rows: kvWidth, cols: d, bf16: self.k},
		{name: "decoder.final_self.v", span: layout.selfV, rows: kvWidth, cols: d, bf16: self.v},
	}
}

func nextParameter(cursor *int, size int) parameterSpan {
	span := parameterSpan{start: *cursor, end: *cursor + size}
	*cursor = span.end
	return span
}

func newTrainingLayout(model *Model) trainingLayout {
	d := model.Dims.DModel
	qWidth := model.Dims.Heads * model.Dims.HeadDim
	kvWidth := model.Dims.KVHeads * model.Dims.HeadDim
	cursor := 0
	layout := trainingLayout{
		finalNorm:     nextParameter(&cursor, d),
		rawGate:       nextParameter(&cursor, 1),
		output:        nextParameter(&cursor, d*qWidth),
		inputNorm:     nextParameter(&cursor, d),
		qNorm:         nextParameter(&cursor, model.Dims.HeadDim),
		kNorm:         nextParameter(&cursor, model.Dims.HeadDim),
		qProjection:   nextParameter(&cursor, qWidth*d),
		kProjection:   nextParameter(&cursor, kvWidth*d),
		vProjection:   nextParameter(&cursor, kvWidth*d),
		selfRawGate:   nextParameter(&cursor, 1),
		selfOutput:    nextParameter(&cursor, d*qWidth),
		selfInputNorm: nextParameter(&cursor, d),
		selfQNorm:     nextParameter(&cursor, model.Dims.HeadDim),
		selfKNorm:     nextParameter(&cursor, model.Dims.HeadDim),
		selfQ:         nextParameter(&cursor, qWidth*d),
		selfK:         nextParameter(&cursor, kvWidth*d),
		selfV:         nextParameter(&cursor, kvWidth*d),
	}
	layout.count = cursor
	return layout
}

// NewTrainer binds declared decoder parameters to compiled execution and Muon.
func NewTrainer(model *Model, steps int, baseLR, momentum float64) (*Trainer, error) {
	if model == nil || steps <= 0 {
		return nil, errors.New("seq2seq: invalid trainer")
	}
	layout := newTrainingLayout(model)
	bindings := trainingParameterBindings(model, layout)
	groups := make([]optimizer.GroupSpec, len(bindings))
	parameters := make([]trainingprogram.ParameterSpec, len(bindings))
	for index, binding := range bindings {
		groups[index] = optimizer.GroupSpec{Name: binding.name, Start: binding.span.start, End: binding.span.end, Rows: binding.rows, Cols: binding.cols}
		parameters[index] = trainingprogram.ParameterSpec{Name: binding.name, Rows: binding.rows, Cols: binding.cols, Trainable: true}
	}
	plan, err := optimizer.CompilePlan(layout.count, groups)
	if err != nil {
		return nil, err
	}
	program, err := trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Operators: []trainingprogram.OperatorSpec{
			{ID: trainingForward, Phase: trainingprogram.PhaseForward},
			{ID: trainingBackward, Phase: trainingprogram.PhaseBackward},
			{ID: trainingMuon, Phase: trainingprogram.PhaseOptimize},
		},
		Parameters: parameters,
		Optimizer:  plan,
	})
	if err != nil {
		return nil, err
	}
	if baseLR <= 0 {
		baseLR = optimizer.DeriveBaseLR(layout.count)
	}
	weights := make([]float32, layout.count)
	for _, binding := range bindings {
		binding.load(weights)
	}
	gradients := make([]float32, layout.count)
	muon, err := newTrainingOptimizer(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: baseLR, Momentum: momentum, Steps: steps, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, err
	}
	trainer := &Trainer{
		model: model, layout: layout, bindings: bindings, program: program, weights: weights, gradients: gradients, optimizer: muon,
	}
	execution, err := trainingprogram.Bind(program, []trainingprogram.Binding[trainingStep]{
		{Operator: trainingForward, Execute: trainer.forward},
		{Operator: trainingBackward, Execute: trainer.backward},
		{Operator: trainingMuon, Execute: trainer.optimize},
	})
	if err != nil {
		_ = muon.Close()
		return nil, err
	}
	trainer.execution = execution
	return trainer, nil
}

func copyBF16ToFloat32(dst []float32, src []uint16) {
	for index, word := range src {
		dst[index] = dtype.BF16ToFloat32(word)
	}
}

func (t *Trainer) Program() trainingprogram.TrainingProgram { return t.program }

func (t *Trainer) Close() error {
	if t == nil || t.optimizer == nil {
		return nil
	}
	return t.optimizer.Close()
}

// Step executes the compiled forward/backward/optimize sequence once.
func (t *Trainer) Step(pair TrainingPair) (float64, error) {
	if t == nil || t.optimizer == nil {
		return 0, errors.New("seq2seq: trainer unavailable")
	}
	state := trainingStep{pair: pair}
	if err := t.execution.RunPhases(&state,
		trainingprogram.PhaseForward, trainingprogram.PhaseBackward, trainingprogram.PhaseOptimize,
	); err != nil {
		return 0, err
	}
	return state.loss, nil
}

func (t *Trainer) forward(state *trainingStep) error {
	if len(state.pair.Source) == 0 || len(state.pair.DecoderInput) == 0 || len(state.pair.DecoderInput) != len(state.pair.Targets) {
		return errors.New("invalid training pair")
	}
	memory, err := t.model.Encode(state.pair.Source)
	if err != nil {
		return err
	}
	state.hidden, err = t.model.decodeHiddenFullTrace(memory, len(state.pair.Source), state.pair.DecoderInput, &state.trace)
	if err != nil {
		return err
	}
	rows, d, vocab := len(state.pair.Targets), t.model.Dims.DModel, t.model.Dims.Vocab
	state.normed = make([]float32, rows*d)
	state.logits = make([]float32, rows*vocab)
	hostmath.RMSNormInto(state.normed, state.hidden, t.model.decFinalNorm, rows, d, t.model.Dims.RMSEps)
	hostmath.LinearBF16(state.logits, state.normed, t.model.embed, rows, d, vocab)
	return nil
}

func (t *Trainer) backward(state *trainingStep) error {
	rows, d, vocab := len(state.pair.Targets), t.model.Dims.DModel, t.model.Dims.Vocab
	for _, target := range state.pair.Targets {
		if target < 0 || target >= vocab {
			return fmt.Errorf("training target %d outside vocab %d", target, vocab)
		}
	}
	dLogits := make([]float32, len(state.logits))
	state.loss = hostmath.SoftmaxCrossEntropy(dLogits, state.logits, state.pair.Targets, rows, vocab)
	dNormed := make([]float32, len(state.normed))
	hostmath.LinearBF16BackwardInput(dNormed, dLogits, t.model.embed, rows, d, vocab)
	state.gradient = make([]float32, len(t.gradients))
	dHidden := make([]float32, len(state.hidden))
	hostmath.RMSNormBackward(
		dHidden, state.gradient[t.layout.finalNorm.start:t.layout.finalNorm.end],
		state.hidden, t.model.decFinalNorm, dNormed, rows, d, t.model.Dims.RMSEps, false,
	)
	return t.backwardFinalCross(state, dHidden)
}

func (t *Trainer) backwardFinalCross(state *trainingStep, dHidden []float32) error {
	rows, d := len(state.pair.Targets), t.model.Dims.DModel
	trace := state.trace.finalCross()
	if len(trace.projected) != len(dHidden) {
		return errors.New("final cross-attention trace differs")
	}
	var gateGradient float64
	for index, gradient := range dHidden {
		gateGradient += float64(gradient) * float64(trace.projected[index])
	}
	block := &t.model.decoderCross[len(t.model.decoderCross)-1]
	gate := block.gate
	state.gradient[t.layout.rawGate.start] = float32(gateGradient) * gate * (1 - gate)
	if len(trace.attention) != rows*t.model.Dims.Heads*t.model.Dims.HeadDim {
		return errors.New("final cross-attention core trace differs")
	}
	dProjected := make([]float32, len(dHidden))
	for index, gradient := range dHidden {
		dProjected[index] = gate * gradient
	}
	state.projectionGradient = state.gradient[t.layout.output.start:t.layout.output.end]
	hostmath.LinearBackward(
		nil, state.projectionGradient, nil, trace.attention, nil, dProjected,
		rows, t.model.Dims.Heads*t.model.Dims.HeadDim, d, false,
	)
	state.attentionQGradient = make([]float32, len(trace.q))
	state.attentionKGradient = make([]float32, len(trace.k))
	state.attentionVGradient = make([]float32, len(trace.v))
	dAttention := make([]float32, len(trace.attention))
	hostmath.LinearBF16BackwardInput(
		dAttention, dProjected, block.o,
		rows, t.model.Dims.Heads*t.model.Dims.HeadDim, d,
	)
	hostmath.MaskedBidirectionalAttentionBackward(
		state.attentionQGradient, state.attentionKGradient, state.attentionVGradient,
		trace.q, trace.k, trace.v, dAttention,
		rows, len(trace.k)/(t.model.Dims.KVHeads*t.model.Dims.HeadDim),
		t.model.Dims.Heads, t.model.Dims.KVHeads, t.model.Dims.HeadDim, nil,
	)
	if err := t.backwardFinalCrossProjections(state, block, dHidden); err != nil {
		return err
	}
	return t.backwardFinalSelfCore(state)
}

func (t *Trainer) backwardFinalCrossProjections(state *trainingStep, block *attnBlock, residualGradient []float32) error {
	dims := t.model.Dims
	trace := state.trace.finalCross()
	rows := len(state.pair.Targets)
	memRows := len(trace.source) / dims.DModel
	qWidth := dims.Heads * dims.HeadDim
	kvWidth := dims.KVHeads * dims.HeadDim
	if len(trace.input) != rows*dims.DModel || len(trace.normed) != rows*dims.DModel ||
		len(trace.qRaw) != rows*qWidth || memRows == 0 || len(trace.kRaw) != memRows*kvWidth {
		return errors.New("final cross-attention projection trace differs")
	}

	dQNorm := make([]float32, len(state.attentionQGradient))
	for index, value := range state.attentionQGradient {
		dQNorm[index] = value * t.model.scoreScale
	}
	dQRaw := make([]float32, len(dQNorm))
	hostmath.RMSNormBackward(
		dQRaw, state.gradient[t.layout.qNorm.start:t.layout.qNorm.end],
		trace.qRaw, block.qNorm, dQNorm,
		rows*dims.Heads, dims.HeadDim, dims.RMSEps, false,
	)
	state.qProjectionGradient = state.gradient[t.layout.qProjection.start:t.layout.qProjection.end]
	hostmath.LinearBackward(
		nil, state.qProjectionGradient, nil, trace.normed, nil, dQRaw,
		rows, dims.DModel, qWidth, false,
	)
	dCrossNormed := make([]float32, len(trace.normed))
	hostmath.LinearBF16BackwardInput(dCrossNormed, dQRaw, block.q, rows, dims.DModel, qWidth)
	dCrossInput := make([]float32, len(dCrossNormed))
	hostmath.RMSNormBackward(
		dCrossInput, state.gradient[t.layout.inputNorm.start:t.layout.inputNorm.end],
		trace.input, block.inNorm, dCrossNormed,
		rows, dims.DModel, dims.RMSEps, false,
	)
	for index, value := range residualGradient {
		dCrossInput[index] += value
	}
	state.finalSelfOutputGradient = dCrossInput

	dKRaw := make([]float32, len(state.attentionKGradient))
	hostmath.RMSNormBackward(
		dKRaw, state.gradient[t.layout.kNorm.start:t.layout.kNorm.end],
		trace.kRaw, block.kNorm, state.attentionKGradient,
		memRows*dims.KVHeads, dims.HeadDim, dims.RMSEps, false,
	)
	state.kProjectionGradient = state.gradient[t.layout.kProjection.start:t.layout.kProjection.end]
	state.vProjectionGradient = state.gradient[t.layout.vProjection.start:t.layout.vProjection.end]
	hostmath.LinearBackward(
		nil, state.kProjectionGradient, nil, trace.source, nil, dKRaw,
		memRows, dims.DModel, kvWidth, false,
	)
	hostmath.LinearBackward(
		nil, state.vProjectionGradient, nil, trace.source, nil, state.attentionVGradient,
		memRows, dims.DModel, kvWidth, false,
	)
	return nil
}

func (t *Trainer) backwardFinalSelfCore(state *trainingStep) error {
	dims := t.model.Dims
	rows := len(state.pair.Targets)
	trace := state.trace.finalSelf()
	if len(state.finalSelfOutputGradient) != rows*dims.DModel || len(trace.projected) != rows*dims.DModel ||
		len(trace.attention) != rows*dims.Heads*dims.HeadDim {
		return errors.New("final self-attention trace differs")
	}
	block := &t.model.decoderSelf[len(t.model.decoderSelf)-1]
	var gateGradient float64
	for index, gradient := range state.finalSelfOutputGradient {
		gateGradient += float64(gradient) * float64(trace.projected[index])
	}
	state.selfRawGateGradient = float32(gateGradient) * block.gate * (1 - block.gate)
	state.gradient[t.layout.selfRawGate.start] = state.selfRawGateGradient
	dProjected := make([]float32, len(state.finalSelfOutputGradient))
	for index, gradient := range state.finalSelfOutputGradient {
		dProjected[index] = block.gate * gradient
	}
	state.selfOutputGradient = state.gradient[t.layout.selfOutput.start:t.layout.selfOutput.end]
	hostmath.LinearBackward(
		nil, state.selfOutputGradient, nil, trace.attention, nil, dProjected,
		rows, dims.Heads*dims.HeadDim, dims.DModel, false,
	)
	dAttention := make([]float32, len(trace.attention))
	hostmath.LinearBF16BackwardInput(
		dAttention, dProjected, block.o,
		rows, dims.Heads*dims.HeadDim, dims.DModel,
	)
	state.selfAttentionQGradient = make([]float32, len(trace.q))
	state.selfAttentionKGradient = make([]float32, len(trace.k))
	state.selfAttentionVGradient = make([]float32, len(trace.v))
	hostmath.CausalAttentionBackward(
		state.selfAttentionQGradient, state.selfAttentionKGradient, state.selfAttentionVGradient,
		trace.q, trace.k, trace.v, dAttention,
		rows, dims.Heads, dims.KVHeads, dims.HeadDim,
	)
	return t.backwardFinalSelfProjections(state, block)
}

func (t *Trainer) backwardFinalSelfProjections(state *trainingStep, block *attnBlock) error {
	dims := t.model.Dims
	rows := len(state.pair.Targets)
	qWidth := dims.Heads * dims.HeadDim
	kvWidth := dims.KVHeads * dims.HeadDim
	trace := state.trace.finalSelf()
	if len(trace.input) != rows*dims.DModel || len(trace.normed) != rows*dims.DModel ||
		len(trace.qRaw) != rows*qWidth || len(trace.kRaw) != rows*kvWidth {
		return errors.New("final self-attention projection trace differs")
	}
	dQNorm := append([]float32(nil), state.selfAttentionQGradient...)
	for index := range dQNorm {
		dQNorm[index] *= t.model.scoreScale
	}
	for row := range rows {
		for head := range dims.Heads {
			hostmath.RotaryHalfBackward(dQNorm[(row*dims.Heads+head)*dims.HeadDim:(row*dims.Heads+head+1)*dims.HeadDim], t.model.invFreq, row)
		}
	}
	dKNorm := append([]float32(nil), state.selfAttentionKGradient...)
	for row := range rows {
		for head := range dims.KVHeads {
			hostmath.RotaryHalfBackward(dKNorm[(row*dims.KVHeads+head)*dims.HeadDim:(row*dims.KVHeads+head+1)*dims.HeadDim], t.model.invFreq, row)
		}
	}
	dQRaw := make([]float32, len(dQNorm))
	dKRaw := make([]float32, len(dKNorm))
	hostmath.RMSNormBackward(
		dQRaw, state.gradient[t.layout.selfQNorm.start:t.layout.selfQNorm.end],
		trace.qRaw, block.qNorm, dQNorm,
		rows*dims.Heads, dims.HeadDim, dims.RMSEps, false,
	)
	hostmath.RMSNormBackward(
		dKRaw, state.gradient[t.layout.selfKNorm.start:t.layout.selfKNorm.end],
		trace.kRaw, block.kNorm, dKNorm,
		rows*dims.KVHeads, dims.HeadDim, dims.RMSEps, false,
	)
	state.selfQGradient = state.gradient[t.layout.selfQ.start:t.layout.selfQ.end]
	state.selfKGradient = state.gradient[t.layout.selfK.start:t.layout.selfK.end]
	state.selfVGradient = state.gradient[t.layout.selfV.start:t.layout.selfV.end]
	hostmath.LinearBackward(nil, state.selfQGradient, nil, trace.normed, nil, dQRaw, rows, dims.DModel, qWidth, false)
	hostmath.LinearBackward(nil, state.selfKGradient, nil, trace.normed, nil, dKRaw, rows, dims.DModel, kvWidth, false)
	hostmath.LinearBackward(nil, state.selfVGradient, nil, trace.normed, nil, state.selfAttentionVGradient, rows, dims.DModel, kvWidth, false)
	dNormed := make([]float32, len(trace.normed))
	scratch := make([]float32, len(dNormed))
	hostmath.LinearBF16BackwardInput(dNormed, dQRaw, block.q, rows, dims.DModel, qWidth)
	hostmath.LinearBF16BackwardInput(scratch, dKRaw, block.k, rows, dims.DModel, kvWidth)
	for index, value := range scratch {
		dNormed[index] += value
	}
	hostmath.LinearBF16BackwardInput(scratch, state.selfAttentionVGradient, block.v, rows, dims.DModel, kvWidth)
	for index, value := range scratch {
		dNormed[index] += value
	}
	dInput := make([]float32, len(dNormed))
	hostmath.RMSNormBackward(
		dInput, state.gradient[t.layout.selfInputNorm.start:t.layout.selfInputNorm.end],
		trace.input, block.inNorm, dNormed, rows, dims.DModel, dims.RMSEps, false,
	)
	for index, value := range state.finalSelfOutputGradient {
		dInput[index] += value
	}
	state.priorDecoderGradient = dInput
	return nil
}

func (t *Trainer) optimize(state *trainingStep) error {
	if len(state.gradient) != len(t.gradients) {
		return errors.New("final-norm gradient differs from parameter plan")
	}
	copy(t.gradients, state.gradient)
	if err := t.optimizer.Step(); err != nil {
		return err
	}
	for _, binding := range t.bindings {
		binding.publish(t.weights)
	}
	return nil
}

func copyFloat32ToBF16(dst []uint16, src []float32) {
	for index, value := range src {
		dst[index] = dtype.Float32ToBF16(value)
	}
}
