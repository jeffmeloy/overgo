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
	pair                           TrainingPair
	memory, hidden, normed, logits []float32
	encoderTrace                   encoderTrainingTrace
	trace                          decoderTrainingTrace
	gradient                       []float32
	projectionGradient             []float32
	qProjectionGradient            []float32
	kProjectionGradient            []float32
	vProjectionGradient            []float32
	attentionQGradient             []float32
	attentionKGradient             []float32
	attentionVGradient             []float32
	selfAttentionQGradient         []float32
	selfAttentionKGradient         []float32
	selfAttentionVGradient         []float32
	selfOutputGradient             []float32
	selfQGradient                  []float32
	selfKGradient                  []float32
	selfVGradient                  []float32
	finalSelfOutputGradient        []float32
	priorDecoderGradient           []float32
	selfRawGateGradient            float32
	penultimateCrossQ              []float32
	penultimateCrossK              []float32
	penultimateCrossV              []float32
	penultimateCrossQProjection    []float32
	penultimateCrossKProjection    []float32
	penultimateCrossVProjection    []float32
	penultimateSelfOutputGradient  []float32
	penultimateSelfQ               []float32
	penultimateSelfK               []float32
	penultimateSelfV               []float32
	penultimateSelfQProjection     []float32
	penultimateSelfKProjection     []float32
	penultimateSelfVProjection     []float32
	previousDecoderGradient        []float32
	decoderInputGradient           []float32
	memoryGradient                 []float32
	encoderInputGradient           []float32
	penultimateSelfRawGate         float32
	penultimateCrossRawGate        float32
	loss                           float64
}

type parameterSpan struct{ start, end int }

type attentionParameterLayout struct {
	rawGate, output, inputNorm parameterSpan
	qNorm, kNorm               parameterSpan
	q, k, v                    parameterSpan
}

type decoderLayerParameterLayout struct{ self, cross attentionParameterLayout }

type trainingLayout struct {
	finalNorm, rawGate, output            parameterSpan
	inputNorm, qNorm, kNorm               parameterSpan
	qProjection, kProjection, vProjection parameterSpan
	selfRawGate, selfOutput               parameterSpan
	selfInputNorm, selfQNorm, selfKNorm   parameterSpan
	selfQ, selfK, selfV                   parameterSpan
	penultimateCrossRawGate               parameterSpan
	penultimateCrossOutput                parameterSpan
	penultimateCrossInputNorm             parameterSpan
	penultimateCrossQNorm                 parameterSpan
	penultimateCrossKNorm                 parameterSpan
	penultimateCrossQ                     parameterSpan
	penultimateCrossK                     parameterSpan
	penultimateCrossV                     parameterSpan
	penultimateSelfRawGate                parameterSpan
	penultimateSelfOutput                 parameterSpan
	penultimateSelfInputNorm              parameterSpan
	penultimateSelfQNorm                  parameterSpan
	penultimateSelfKNorm                  parameterSpan
	penultimateSelfQ                      parameterSpan
	penultimateSelfK                      parameterSpan
	penultimateSelfV                      parameterSpan
	earlierDecoder                        []decoderLayerParameterLayout
	encoderFinalNorm                      parameterSpan
	encoder                               []attentionParameterLayout
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
	penultimateCross := &model.decoderCross[len(model.decoderCross)-2]
	penultimateSelf := &model.decoderSelf[len(model.decoderSelf)-2]
	penultimateName := fmt.Sprintf("decoder.layer%d.cross", len(model.decoderCross)-2)
	penultimateSelfName := fmt.Sprintf("decoder.layer%d.self", len(model.decoderSelf)-2)
	d := model.Dims.DModel
	qWidth := model.Dims.Heads * model.Dims.HeadDim
	kvWidth := model.Dims.KVHeads * model.Dims.HeadDim
	bindings := []trainingParameterBinding{
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
		{name: penultimateName + ".raw_gate", span: layout.penultimateCrossRawGate, rows: 1, cols: 1, scalar: &penultimateCross.rawGate, folded: &penultimateCross.gate},
		{name: penultimateName + ".output", span: layout.penultimateCrossOutput, rows: d, cols: qWidth, bf16: penultimateCross.o},
		{name: penultimateName + ".input_norm", span: layout.penultimateCrossInputNorm, rows: d, cols: 1, f32: penultimateCross.inNorm},
		{name: penultimateName + ".q_norm", span: layout.penultimateCrossQNorm, rows: model.Dims.HeadDim, cols: 1, f32: penultimateCross.qNorm},
		{name: penultimateName + ".k_norm", span: layout.penultimateCrossKNorm, rows: model.Dims.HeadDim, cols: 1, f32: penultimateCross.kNorm},
		{name: penultimateName + ".q", span: layout.penultimateCrossQ, rows: qWidth, cols: d, bf16: penultimateCross.q},
		{name: penultimateName + ".k", span: layout.penultimateCrossK, rows: kvWidth, cols: d, bf16: penultimateCross.k},
		{name: penultimateName + ".v", span: layout.penultimateCrossV, rows: kvWidth, cols: d, bf16: penultimateCross.v},
		{name: penultimateSelfName + ".raw_gate", span: layout.penultimateSelfRawGate, rows: 1, cols: 1, scalar: &penultimateSelf.rawGate, folded: &penultimateSelf.gate},
		{name: penultimateSelfName + ".output", span: layout.penultimateSelfOutput, rows: d, cols: qWidth, bf16: penultimateSelf.o},
		{name: penultimateSelfName + ".input_norm", span: layout.penultimateSelfInputNorm, rows: d, cols: 1, f32: penultimateSelf.inNorm},
		{name: penultimateSelfName + ".q_norm", span: layout.penultimateSelfQNorm, rows: model.Dims.HeadDim, cols: 1, f32: penultimateSelf.qNorm},
		{name: penultimateSelfName + ".k_norm", span: layout.penultimateSelfKNorm, rows: model.Dims.HeadDim, cols: 1, f32: penultimateSelf.kNorm},
		{name: penultimateSelfName + ".q", span: layout.penultimateSelfQ, rows: qWidth, cols: d, bf16: penultimateSelf.q},
		{name: penultimateSelfName + ".k", span: layout.penultimateSelfK, rows: kvWidth, cols: d, bf16: penultimateSelf.k},
		{name: penultimateSelfName + ".v", span: layout.penultimateSelfV, rows: kvWidth, cols: d, bf16: penultimateSelf.v},
	}
	for layer := range layout.earlierDecoder {
		bindings = appendAttentionParameterBindings(bindings, fmt.Sprintf("decoder.layer%d.self", layer), layout.earlierDecoder[layer].self, &model.decoderSelf[layer], d, qWidth, kvWidth)
		bindings = appendAttentionParameterBindings(bindings, fmt.Sprintf("decoder.layer%d.cross", layer), layout.earlierDecoder[layer].cross, &model.decoderCross[layer], d, qWidth, kvWidth)
	}
	bindings = append(bindings, trainingParameterBinding{name: "encoder.final_norm", span: layout.encoderFinalNorm, rows: d, cols: 1, f32: model.encFinalNorm})
	for layer := range layout.encoder {
		bindings = appendAttentionParameterBindings(bindings, fmt.Sprintf("encoder.layer%d.self", layer), layout.encoder[layer], &model.encoder[layer], d, qWidth, kvWidth)
	}
	return bindings
}

func appendAttentionParameterBindings(
	dst []trainingParameterBinding, name string, layout attentionParameterLayout,
	block *attnBlock, d, qWidth, kvWidth int,
) []trainingParameterBinding {
	return append(dst,
		trainingParameterBinding{name: name + ".raw_gate", span: layout.rawGate, rows: 1, cols: 1, scalar: &block.rawGate, folded: &block.gate},
		trainingParameterBinding{name: name + ".output", span: layout.output, rows: d, cols: qWidth, bf16: block.o},
		trainingParameterBinding{name: name + ".input_norm", span: layout.inputNorm, rows: d, cols: 1, f32: block.inNorm},
		trainingParameterBinding{name: name + ".q_norm", span: layout.qNorm, rows: len(block.qNorm), cols: 1, f32: block.qNorm},
		trainingParameterBinding{name: name + ".k_norm", span: layout.kNorm, rows: len(block.kNorm), cols: 1, f32: block.kNorm},
		trainingParameterBinding{name: name + ".q", span: layout.q, rows: qWidth, cols: d, bf16: block.q},
		trainingParameterBinding{name: name + ".k", span: layout.k, rows: kvWidth, cols: d, bf16: block.k},
		trainingParameterBinding{name: name + ".v", span: layout.v, rows: kvWidth, cols: d, bf16: block.v},
	)
}

func nextParameter(cursor *int, size int) parameterSpan {
	span := parameterSpan{start: *cursor, end: *cursor + size}
	*cursor = span.end
	return span
}

func nextAttentionLayout(cursor *int, d, headDim, qWidth, kvWidth int) attentionParameterLayout {
	return attentionParameterLayout{
		rawGate: nextParameter(cursor, 1), output: nextParameter(cursor, d*qWidth),
		inputNorm: nextParameter(cursor, d), qNorm: nextParameter(cursor, headDim), kNorm: nextParameter(cursor, headDim),
		q: nextParameter(cursor, qWidth*d), k: nextParameter(cursor, kvWidth*d), v: nextParameter(cursor, kvWidth*d),
	}
}

func newTrainingLayout(model *Model) trainingLayout {
	d := model.Dims.DModel
	qWidth := model.Dims.Heads * model.Dims.HeadDim
	kvWidth := model.Dims.KVHeads * model.Dims.HeadDim
	cursor := 0
	layout := trainingLayout{
		finalNorm:                 nextParameter(&cursor, d),
		rawGate:                   nextParameter(&cursor, 1),
		output:                    nextParameter(&cursor, d*qWidth),
		inputNorm:                 nextParameter(&cursor, d),
		qNorm:                     nextParameter(&cursor, model.Dims.HeadDim),
		kNorm:                     nextParameter(&cursor, model.Dims.HeadDim),
		qProjection:               nextParameter(&cursor, qWidth*d),
		kProjection:               nextParameter(&cursor, kvWidth*d),
		vProjection:               nextParameter(&cursor, kvWidth*d),
		selfRawGate:               nextParameter(&cursor, 1),
		selfOutput:                nextParameter(&cursor, d*qWidth),
		selfInputNorm:             nextParameter(&cursor, d),
		selfQNorm:                 nextParameter(&cursor, model.Dims.HeadDim),
		selfKNorm:                 nextParameter(&cursor, model.Dims.HeadDim),
		selfQ:                     nextParameter(&cursor, qWidth*d),
		selfK:                     nextParameter(&cursor, kvWidth*d),
		selfV:                     nextParameter(&cursor, kvWidth*d),
		penultimateCrossRawGate:   nextParameter(&cursor, 1),
		penultimateCrossOutput:    nextParameter(&cursor, d*qWidth),
		penultimateCrossInputNorm: nextParameter(&cursor, d),
		penultimateCrossQNorm:     nextParameter(&cursor, model.Dims.HeadDim),
		penultimateCrossKNorm:     nextParameter(&cursor, model.Dims.HeadDim),
		penultimateCrossQ:         nextParameter(&cursor, qWidth*d),
		penultimateCrossK:         nextParameter(&cursor, kvWidth*d),
		penultimateCrossV:         nextParameter(&cursor, kvWidth*d),
		penultimateSelfRawGate:    nextParameter(&cursor, 1),
		penultimateSelfOutput:     nextParameter(&cursor, d*qWidth),
		penultimateSelfInputNorm:  nextParameter(&cursor, d),
		penultimateSelfQNorm:      nextParameter(&cursor, model.Dims.HeadDim),
		penultimateSelfKNorm:      nextParameter(&cursor, model.Dims.HeadDim),
		penultimateSelfQ:          nextParameter(&cursor, qWidth*d),
		penultimateSelfK:          nextParameter(&cursor, kvWidth*d),
		penultimateSelfV:          nextParameter(&cursor, kvWidth*d),
	}
	layout.earlierDecoder = make([]decoderLayerParameterLayout, model.Dims.DecoderLayers-2)
	for layer := range layout.earlierDecoder {
		layout.earlierDecoder[layer] = decoderLayerParameterLayout{
			self:  nextAttentionLayout(&cursor, d, model.Dims.HeadDim, qWidth, kvWidth),
			cross: nextAttentionLayout(&cursor, d, model.Dims.HeadDim, qWidth, kvWidth),
		}
	}
	layout.encoderFinalNorm = nextParameter(&cursor, d)
	layout.encoder = make([]attentionParameterLayout, model.Dims.EncoderLayers)
	for layer := range layout.encoder {
		layout.encoder[layer] = nextAttentionLayout(&cursor, d, model.Dims.HeadDim, qWidth, kvWidth)
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
	var err error
	state.memory, err = t.model.encodeTrace(state.pair.Source, &state.encoderTrace)
	if err != nil {
		return err
	}
	state.hidden, err = t.model.decodeHiddenFullTrace(state.memory, len(state.pair.Source), state.pair.DecoderInput, &state.trace)
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
	rows := len(state.pair.Targets)
	trace := state.trace.finalCross()
	block := &t.model.decoderCross[len(t.model.decoderCross)-1]
	core, err := backwardCrossAttentionCore(trace, block, dHidden, rows, t.model.Dims)
	if err != nil {
		return err
	}
	state.gradient[t.layout.rawGate.start] = core.rawGate
	state.projectionGradient = state.gradient[t.layout.output.start:t.layout.output.end]
	copy(state.projectionGradient, core.output)
	state.attentionQGradient, state.attentionKGradient, state.attentionVGradient = core.q, core.k, core.v
	if err := t.backwardFinalCrossProjections(state, block, dHidden); err != nil {
		return err
	}
	return t.backwardFinalSelfCore(state)
}

type attentionCoreGradient struct {
	rawGate         float32
	output, q, k, v []float32
}

type crossProjectionGradient struct {
	input, source, inputNorm, qNorm, kNorm []float32
	q, k, v                                []float32
}

func backwardCrossAttentionCore(trace *attentionTrainingTrace, block *attnBlock, dOutput []float32, rows int, dims Dims) (attentionCoreGradient, error) {
	if len(trace.projected) != len(dOutput) || len(trace.attention) != rows*dims.Heads*dims.HeadDim {
		return attentionCoreGradient{}, errors.New("cross-attention core trace differs")
	}
	var gateGradient float64
	for index, gradient := range dOutput {
		gateGradient += float64(gradient) * float64(trace.projected[index])
	}
	result := attentionCoreGradient{rawGate: float32(gateGradient) * block.gate * (1 - block.gate)}
	dProjected := make([]float32, len(dOutput))
	for index, gradient := range dOutput {
		dProjected[index] = block.gate * gradient
	}
	result.output = make([]float32, len(block.o))
	hostmath.LinearBackward(nil, result.output, nil, trace.attention, nil, dProjected, rows, dims.Heads*dims.HeadDim, dims.DModel, false)
	dAttention := make([]float32, len(trace.attention))
	hostmath.LinearBF16BackwardInput(dAttention, dProjected, block.o, rows, dims.Heads*dims.HeadDim, dims.DModel)
	result.q = make([]float32, len(trace.q))
	result.k = make([]float32, len(trace.k))
	result.v = make([]float32, len(trace.v))
	hostmath.MaskedBidirectionalAttentionBackward(
		result.q, result.k, result.v, trace.q, trace.k, trace.v, dAttention,
		rows, len(trace.k)/(dims.KVHeads*dims.HeadDim), dims.Heads, dims.KVHeads, dims.HeadDim, nil,
	)
	return result, nil
}

func backwardCrossAttentionProjections(
	trace *attentionTrainingTrace, block *attnBlock, core attentionCoreGradient,
	residualGradient []float32, rows int, dims Dims, scoreScale float32,
) (crossProjectionGradient, error) {
	memRows := len(trace.source) / dims.DModel
	qWidth := dims.Heads * dims.HeadDim
	kvWidth := dims.KVHeads * dims.HeadDim
	if len(trace.input) != rows*dims.DModel || len(trace.normed) != rows*dims.DModel ||
		len(trace.qRaw) != rows*qWidth || memRows == 0 || len(trace.kRaw) != memRows*kvWidth {
		return crossProjectionGradient{}, errors.New("cross-attention projection trace differs")
	}
	result := crossProjectionGradient{
		inputNorm: make([]float32, dims.DModel), qNorm: make([]float32, dims.HeadDim), kNorm: make([]float32, dims.HeadDim),
		q: make([]float32, len(block.q)), k: make([]float32, len(block.k)), v: make([]float32, len(block.v)),
	}
	dQNorm := make([]float32, len(core.q))
	for index, value := range core.q {
		dQNorm[index] = value * scoreScale
	}
	dQRaw := make([]float32, len(dQNorm))
	hostmath.RMSNormBackward(
		dQRaw, result.qNorm, trace.qRaw, block.qNorm, dQNorm,
		rows*dims.Heads, dims.HeadDim, dims.RMSEps, false,
	)
	hostmath.LinearBackward(nil, result.q, nil, trace.normed, nil, dQRaw, rows, dims.DModel, qWidth, false)
	dCrossNormed := make([]float32, len(trace.normed))
	hostmath.LinearBF16BackwardInput(dCrossNormed, dQRaw, block.q, rows, dims.DModel, qWidth)
	result.input = make([]float32, len(dCrossNormed))
	hostmath.RMSNormBackward(
		result.input, result.inputNorm, trace.input, block.inNorm, dCrossNormed,
		rows, dims.DModel, dims.RMSEps, false,
	)
	for index, value := range residualGradient {
		result.input[index] += value
	}
	dKRaw := make([]float32, len(core.k))
	hostmath.RMSNormBackward(
		dKRaw, result.kNorm, trace.kRaw, block.kNorm, core.k,
		memRows*dims.KVHeads, dims.HeadDim, dims.RMSEps, false,
	)
	hostmath.LinearBackward(nil, result.k, nil, trace.source, nil, dKRaw, memRows, dims.DModel, kvWidth, false)
	hostmath.LinearBackward(nil, result.v, nil, trace.source, nil, core.v, memRows, dims.DModel, kvWidth, false)
	result.source = make([]float32, len(trace.source))
	sourceScratch := make([]float32, len(trace.source))
	hostmath.LinearBF16BackwardInput(result.source, dKRaw, block.k, memRows, dims.DModel, kvWidth)
	hostmath.LinearBF16BackwardInput(sourceScratch, core.v, block.v, memRows, dims.DModel, kvWidth)
	for index, value := range sourceScratch {
		result.source[index] += value
	}
	return result, nil
}

func accumulateGradient(dst *[]float32, src []float32) error {
	if len(src) == 0 {
		return errors.New("seq2seq: empty accumulated gradient")
	}
	if len(*dst) == 0 {
		*dst = append([]float32(nil), src...)
		return nil
	}
	if len(*dst) != len(src) {
		return errors.New("seq2seq: accumulated gradient geometry differs")
	}
	for index, value := range src {
		(*dst)[index] += value
	}
	return nil
}

func (t *Trainer) backwardFinalCrossProjections(state *trainingStep, block *attnBlock, residualGradient []float32) error {
	result, err := backwardCrossAttentionProjections(
		state.trace.finalCross(), block,
		attentionCoreGradient{q: state.attentionQGradient, k: state.attentionKGradient, v: state.attentionVGradient},
		residualGradient, len(state.pair.Targets), t.model.Dims, t.model.scoreScale,
	)
	if err != nil {
		return err
	}
	copy(state.gradient[t.layout.inputNorm.start:t.layout.inputNorm.end], result.inputNorm)
	copy(state.gradient[t.layout.qNorm.start:t.layout.qNorm.end], result.qNorm)
	copy(state.gradient[t.layout.kNorm.start:t.layout.kNorm.end], result.kNorm)
	state.qProjectionGradient = state.gradient[t.layout.qProjection.start:t.layout.qProjection.end]
	state.kProjectionGradient = state.gradient[t.layout.kProjection.start:t.layout.kProjection.end]
	state.vProjectionGradient = state.gradient[t.layout.vProjection.start:t.layout.vProjection.end]
	copy(state.qProjectionGradient, result.q)
	copy(state.kProjectionGradient, result.k)
	copy(state.vProjectionGradient, result.v)
	state.finalSelfOutputGradient = result.input
	return accumulateGradient(&state.memoryGradient, result.source)
}

func (t *Trainer) backwardFinalSelfCore(state *trainingStep) error {
	dims := t.model.Dims
	rows := len(state.pair.Targets)
	trace := state.trace.finalSelf()
	block := &t.model.decoderSelf[len(t.model.decoderSelf)-1]
	core, err := backwardCausalAttentionCore(trace, block, state.finalSelfOutputGradient, rows, dims)
	if err != nil {
		return err
	}
	state.selfRawGateGradient = core.rawGate
	state.gradient[t.layout.selfRawGate.start] = state.selfRawGateGradient
	state.selfOutputGradient = state.gradient[t.layout.selfOutput.start:t.layout.selfOutput.end]
	copy(state.selfOutputGradient, core.output)
	state.selfAttentionQGradient, state.selfAttentionKGradient, state.selfAttentionVGradient = core.q, core.k, core.v
	return t.backwardFinalSelfProjections(state, block)
}

func backwardCausalAttentionCore(trace *attentionTrainingTrace, block *attnBlock, dOutput []float32, rows int, dims Dims) (attentionCoreGradient, error) {
	if len(dOutput) != rows*dims.DModel || len(trace.projected) != len(dOutput) || len(trace.attention) != rows*dims.Heads*dims.HeadDim {
		return attentionCoreGradient{}, errors.New("causal-attention core trace differs")
	}
	var gateGradient float64
	for index, gradient := range dOutput {
		gateGradient += float64(gradient) * float64(trace.projected[index])
	}
	result := attentionCoreGradient{rawGate: float32(gateGradient) * block.gate * (1 - block.gate)}
	dProjected := make([]float32, len(dOutput))
	for index, gradient := range dOutput {
		dProjected[index] = block.gate * gradient
	}
	result.output = make([]float32, len(block.o))
	hostmath.LinearBackward(nil, result.output, nil, trace.attention, nil, dProjected, rows, dims.Heads*dims.HeadDim, dims.DModel, false)
	dAttention := make([]float32, len(trace.attention))
	hostmath.LinearBF16BackwardInput(dAttention, dProjected, block.o, rows, dims.Heads*dims.HeadDim, dims.DModel)
	result.q = make([]float32, len(trace.q))
	result.k = make([]float32, len(trace.k))
	result.v = make([]float32, len(trace.v))
	hostmath.CausalAttentionBackward(result.q, result.k, result.v, trace.q, trace.k, trace.v, dAttention, rows, dims.Heads, dims.KVHeads, dims.HeadDim)
	return result, nil
}

func backwardCausalAttentionProjections(
	trace *attentionTrainingTrace, block *attnBlock, core attentionCoreGradient,
	residualGradient []float32, rows int, dims Dims, scoreScale float32, invFreq []float64,
) (crossProjectionGradient, error) {
	qWidth := dims.Heads * dims.HeadDim
	kvWidth := dims.KVHeads * dims.HeadDim
	if len(trace.input) != rows*dims.DModel || len(trace.normed) != rows*dims.DModel ||
		len(trace.qRaw) != rows*qWidth || len(trace.kRaw) != rows*kvWidth {
		return crossProjectionGradient{}, errors.New("causal-attention projection trace differs")
	}
	result := crossProjectionGradient{
		inputNorm: make([]float32, dims.DModel), qNorm: make([]float32, dims.HeadDim), kNorm: make([]float32, dims.HeadDim),
		q: make([]float32, len(block.q)), k: make([]float32, len(block.k)), v: make([]float32, len(block.v)),
	}
	dQNorm := append([]float32(nil), core.q...)
	for index := range dQNorm {
		dQNorm[index] *= scoreScale
	}
	for row := range rows {
		for head := range dims.Heads {
			hostmath.RotaryHalfBackward(dQNorm[(row*dims.Heads+head)*dims.HeadDim:(row*dims.Heads+head+1)*dims.HeadDim], invFreq, row)
		}
	}
	dKNorm := append([]float32(nil), core.k...)
	for row := range rows {
		for head := range dims.KVHeads {
			hostmath.RotaryHalfBackward(dKNorm[(row*dims.KVHeads+head)*dims.HeadDim:(row*dims.KVHeads+head+1)*dims.HeadDim], invFreq, row)
		}
	}
	dQRaw := make([]float32, len(dQNorm))
	dKRaw := make([]float32, len(dKNorm))
	hostmath.RMSNormBackward(
		dQRaw, result.qNorm,
		trace.qRaw, block.qNorm, dQNorm,
		rows*dims.Heads, dims.HeadDim, dims.RMSEps, false,
	)
	hostmath.RMSNormBackward(
		dKRaw, result.kNorm,
		trace.kRaw, block.kNorm, dKNorm,
		rows*dims.KVHeads, dims.HeadDim, dims.RMSEps, false,
	)
	hostmath.LinearBackward(nil, result.q, nil, trace.normed, nil, dQRaw, rows, dims.DModel, qWidth, false)
	hostmath.LinearBackward(nil, result.k, nil, trace.normed, nil, dKRaw, rows, dims.DModel, kvWidth, false)
	hostmath.LinearBackward(nil, result.v, nil, trace.normed, nil, core.v, rows, dims.DModel, kvWidth, false)
	dNormed := make([]float32, len(trace.normed))
	scratch := make([]float32, len(dNormed))
	hostmath.LinearBF16BackwardInput(dNormed, dQRaw, block.q, rows, dims.DModel, qWidth)
	hostmath.LinearBF16BackwardInput(scratch, dKRaw, block.k, rows, dims.DModel, kvWidth)
	for index, value := range scratch {
		dNormed[index] += value
	}
	hostmath.LinearBF16BackwardInput(scratch, core.v, block.v, rows, dims.DModel, kvWidth)
	for index, value := range scratch {
		dNormed[index] += value
	}
	result.input = make([]float32, len(dNormed))
	hostmath.RMSNormBackward(
		result.input, result.inputNorm,
		trace.input, block.inNorm, dNormed, rows, dims.DModel, dims.RMSEps, false,
	)
	for index, value := range residualGradient {
		result.input[index] += value
	}
	return result, nil
}

func (t *Trainer) backwardFinalSelfProjections(state *trainingStep, block *attnBlock) error {
	result, err := backwardCausalAttentionProjections(
		state.trace.finalSelf(), block,
		attentionCoreGradient{q: state.selfAttentionQGradient, k: state.selfAttentionKGradient, v: state.selfAttentionVGradient},
		state.finalSelfOutputGradient, len(state.pair.Targets), t.model.Dims, t.model.scoreScale, t.model.invFreq,
	)
	if err != nil {
		return err
	}
	copy(state.gradient[t.layout.selfInputNorm.start:t.layout.selfInputNorm.end], result.inputNorm)
	copy(state.gradient[t.layout.selfQNorm.start:t.layout.selfQNorm.end], result.qNorm)
	copy(state.gradient[t.layout.selfKNorm.start:t.layout.selfKNorm.end], result.kNorm)
	state.selfQGradient = state.gradient[t.layout.selfQ.start:t.layout.selfQ.end]
	state.selfKGradient = state.gradient[t.layout.selfK.start:t.layout.selfK.end]
	state.selfVGradient = state.gradient[t.layout.selfV.start:t.layout.selfV.end]
	copy(state.selfQGradient, result.q)
	copy(state.selfKGradient, result.k)
	copy(state.selfVGradient, result.v)
	state.priorDecoderGradient = result.input
	return t.backwardPenultimateCrossCore(state)
}

func (t *Trainer) backwardPenultimateCrossCore(state *trainingStep) error {
	layer := len(t.model.decoderCross) - 2
	if layer < 0 {
		return nil
	}
	core, err := backwardCrossAttentionCore(
		&state.trace.layers[layer].cross, &t.model.decoderCross[layer],
		state.priorDecoderGradient, len(state.pair.Targets), t.model.Dims,
	)
	if err != nil {
		return err
	}
	state.penultimateCrossQ, state.penultimateCrossK, state.penultimateCrossV = core.q, core.k, core.v
	state.penultimateCrossRawGate = core.rawGate
	state.gradient[t.layout.penultimateCrossRawGate.start] = core.rawGate
	copy(state.gradient[t.layout.penultimateCrossOutput.start:t.layout.penultimateCrossOutput.end], core.output)
	projections, err := backwardCrossAttentionProjections(
		&state.trace.layers[layer].cross, &t.model.decoderCross[layer], core,
		state.priorDecoderGradient, len(state.pair.Targets), t.model.Dims, t.model.scoreScale,
	)
	if err != nil {
		return err
	}
	copy(state.gradient[t.layout.penultimateCrossInputNorm.start:t.layout.penultimateCrossInputNorm.end], projections.inputNorm)
	copy(state.gradient[t.layout.penultimateCrossQNorm.start:t.layout.penultimateCrossQNorm.end], projections.qNorm)
	copy(state.gradient[t.layout.penultimateCrossKNorm.start:t.layout.penultimateCrossKNorm.end], projections.kNorm)
	state.penultimateCrossQProjection = state.gradient[t.layout.penultimateCrossQ.start:t.layout.penultimateCrossQ.end]
	state.penultimateCrossKProjection = state.gradient[t.layout.penultimateCrossK.start:t.layout.penultimateCrossK.end]
	state.penultimateCrossVProjection = state.gradient[t.layout.penultimateCrossV.start:t.layout.penultimateCrossV.end]
	copy(state.penultimateCrossQProjection, projections.q)
	copy(state.penultimateCrossKProjection, projections.k)
	copy(state.penultimateCrossVProjection, projections.v)
	state.penultimateSelfOutputGradient = projections.input
	if err := accumulateGradient(&state.memoryGradient, projections.source); err != nil {
		return err
	}
	return t.backwardPenultimateSelfCore(state)
}

func (t *Trainer) backwardPenultimateSelfCore(state *trainingStep) error {
	layer := len(t.model.decoderSelf) - 2
	if layer < 0 {
		return nil
	}
	core, err := backwardCausalAttentionCore(
		&state.trace.layers[layer].self, &t.model.decoderSelf[layer],
		state.penultimateSelfOutputGradient, len(state.pair.Targets), t.model.Dims,
	)
	if err != nil {
		return err
	}
	state.penultimateSelfQ, state.penultimateSelfK, state.penultimateSelfV = core.q, core.k, core.v
	state.penultimateSelfRawGate = core.rawGate
	state.gradient[t.layout.penultimateSelfRawGate.start] = core.rawGate
	copy(state.gradient[t.layout.penultimateSelfOutput.start:t.layout.penultimateSelfOutput.end], core.output)
	projections, err := backwardCausalAttentionProjections(
		&state.trace.layers[layer].self, &t.model.decoderSelf[layer], core,
		state.penultimateSelfOutputGradient, len(state.pair.Targets), t.model.Dims, t.model.scoreScale, t.model.invFreq,
	)
	if err != nil {
		return err
	}
	copy(state.gradient[t.layout.penultimateSelfInputNorm.start:t.layout.penultimateSelfInputNorm.end], projections.inputNorm)
	copy(state.gradient[t.layout.penultimateSelfQNorm.start:t.layout.penultimateSelfQNorm.end], projections.qNorm)
	copy(state.gradient[t.layout.penultimateSelfKNorm.start:t.layout.penultimateSelfKNorm.end], projections.kNorm)
	state.penultimateSelfQProjection = state.gradient[t.layout.penultimateSelfQ.start:t.layout.penultimateSelfQ.end]
	state.penultimateSelfKProjection = state.gradient[t.layout.penultimateSelfK.start:t.layout.penultimateSelfK.end]
	state.penultimateSelfVProjection = state.gradient[t.layout.penultimateSelfV.start:t.layout.penultimateSelfV.end]
	copy(state.penultimateSelfQProjection, projections.q)
	copy(state.penultimateSelfKProjection, projections.k)
	copy(state.penultimateSelfVProjection, projections.v)
	state.previousDecoderGradient = projections.input
	return t.backwardEarlierDecoderLayers(state)
}

func writeAttentionGradients(dst []float32, layout attentionParameterLayout, core attentionCoreGradient, projections crossProjectionGradient) {
	dst[layout.rawGate.start] = core.rawGate
	copy(dst[layout.output.start:layout.output.end], core.output)
	copy(dst[layout.inputNorm.start:layout.inputNorm.end], projections.inputNorm)
	copy(dst[layout.qNorm.start:layout.qNorm.end], projections.qNorm)
	copy(dst[layout.kNorm.start:layout.kNorm.end], projections.kNorm)
	copy(dst[layout.q.start:layout.q.end], projections.q)
	copy(dst[layout.k.start:layout.k.end], projections.k)
	copy(dst[layout.v.start:layout.v.end], projections.v)
}

func (t *Trainer) backwardEarlierDecoderLayers(state *trainingStep) error {
	gradient := state.previousDecoderGradient
	rows := len(state.pair.Targets)
	for layer := len(t.layout.earlierDecoder) - 1; layer >= 0; layer-- {
		layout := t.layout.earlierDecoder[layer]
		crossTrace := &state.trace.layers[layer].cross
		crossBlock := &t.model.decoderCross[layer]
		crossCore, err := backwardCrossAttentionCore(crossTrace, crossBlock, gradient, rows, t.model.Dims)
		if err != nil {
			return err
		}
		crossProjections, err := backwardCrossAttentionProjections(
			crossTrace, crossBlock, crossCore, gradient, rows, t.model.Dims, t.model.scoreScale,
		)
		if err != nil {
			return err
		}
		writeAttentionGradients(state.gradient, layout.cross, crossCore, crossProjections)
		if err := accumulateGradient(&state.memoryGradient, crossProjections.source); err != nil {
			return err
		}
		gradient = crossProjections.input

		selfTrace := &state.trace.layers[layer].self
		selfBlock := &t.model.decoderSelf[layer]
		selfCore, err := backwardCausalAttentionCore(selfTrace, selfBlock, gradient, rows, t.model.Dims)
		if err != nil {
			return err
		}
		selfProjections, err := backwardCausalAttentionProjections(
			selfTrace, selfBlock, selfCore, gradient, rows, t.model.Dims, t.model.scoreScale, t.model.invFreq,
		)
		if err != nil {
			return err
		}
		writeAttentionGradients(state.gradient, layout.self, selfCore, selfProjections)
		gradient = selfProjections.input
	}
	state.decoderInputGradient = gradient
	return t.backwardEncoder(state)
}

func (t *Trainer) backwardEncoder(state *trainingStep) error {
	rows, d := len(state.pair.Source), t.model.Dims.DModel
	if len(state.memoryGradient) != len(state.memory) || len(state.encoderTrace.hidden) != rows*d {
		return errors.New("seq2seq: encoder gradient trace differs")
	}
	gradient := make([]float32, len(state.encoderTrace.hidden))
	hostmath.RMSNormBackward(
		gradient, state.gradient[t.layout.encoderFinalNorm.start:t.layout.encoderFinalNorm.end],
		state.encoderTrace.hidden, t.model.encFinalNorm, state.memoryGradient,
		rows, d, t.model.Dims.RMSEps, false,
	)
	for layer := len(t.layout.encoder) - 1; layer >= 0; layer-- {
		trace := &state.encoderTrace.layers[layer]
		block := &t.model.encoder[layer]
		core, err := backwardCrossAttentionCore(trace, block, gradient, rows, t.model.Dims)
		if err != nil {
			return err
		}
		projections, err := backwardCausalAttentionProjections(
			trace, block, core, gradient, rows, t.model.Dims, t.model.scoreScale, t.model.invFreq,
		)
		if err != nil {
			return err
		}
		writeAttentionGradients(state.gradient, t.layout.encoder[layer], core, projections)
		gradient = projections.input
	}
	state.encoderInputGradient = gradient
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
