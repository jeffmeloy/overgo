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
	pair                   TrainingPair
	hidden, normed, logits []float32
	trace                  decoderTrainingTrace
	gradient               []float32
	projectionGradient     []float32
	qProjectionGradient    []float32
	kProjectionGradient    []float32
	vProjectionGradient    []float32
	attentionQGradient     []float32
	attentionKGradient     []float32
	attentionVGradient     []float32
	loss                   float64
}

type parameterSpan struct{ start, end int }

type trainingLayout struct {
	finalNorm, rawGate, output            parameterSpan
	inputNorm, qNorm, kNorm               parameterSpan
	qProjection, kProjection, vProjection parameterSpan
	count                                 int
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
		finalNorm:   nextParameter(&cursor, d),
		rawGate:     nextParameter(&cursor, 1),
		output:      nextParameter(&cursor, d*qWidth),
		inputNorm:   nextParameter(&cursor, d),
		qNorm:       nextParameter(&cursor, model.Dims.HeadDim),
		kNorm:       nextParameter(&cursor, model.Dims.HeadDim),
		qProjection: nextParameter(&cursor, qWidth*d),
		kProjection: nextParameter(&cursor, kvWidth*d),
		vProjection: nextParameter(&cursor, kvWidth*d),
	}
	layout.count = cursor
	return layout
}

// NewTrainer binds declared decoder parameters to compiled execution and Muon.
func NewTrainer(model *Model, steps int, baseLR, momentum float64) (*Trainer, error) {
	if model == nil || steps <= 0 {
		return nil, errors.New("seq2seq: invalid trainer")
	}
	d, width := model.Dims.DModel, model.Dims.Heads*model.Dims.HeadDim
	kvWidth := model.Dims.KVHeads * model.Dims.HeadDim
	layout := newTrainingLayout(model)
	plan, err := optimizer.CompilePlan(layout.count, []optimizer.GroupSpec{
		{Name: "decoder.final_norm", Start: layout.finalNorm.start, End: layout.finalNorm.end, Rows: d, Cols: 1},
		{Name: "decoder.final_cross.raw_gate", Start: layout.rawGate.start, End: layout.rawGate.end, Rows: 1, Cols: 1},
		{Name: "decoder.final_cross.output", Start: layout.output.start, End: layout.output.end, Rows: d, Cols: width},
		{Name: "decoder.final_cross.input_norm", Start: layout.inputNorm.start, End: layout.inputNorm.end, Rows: d, Cols: 1},
		{Name: "decoder.final_cross.q_norm", Start: layout.qNorm.start, End: layout.qNorm.end, Rows: model.Dims.HeadDim, Cols: 1},
		{Name: "decoder.final_cross.k_norm", Start: layout.kNorm.start, End: layout.kNorm.end, Rows: model.Dims.HeadDim, Cols: 1},
		{Name: "decoder.final_cross.q", Start: layout.qProjection.start, End: layout.qProjection.end, Rows: width, Cols: d},
		{Name: "decoder.final_cross.k", Start: layout.kProjection.start, End: layout.kProjection.end, Rows: kvWidth, Cols: d},
		{Name: "decoder.final_cross.v", Start: layout.vProjection.start, End: layout.vProjection.end, Rows: kvWidth, Cols: d},
	})
	if err != nil {
		return nil, err
	}
	program, err := trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Operators: []trainingprogram.OperatorSpec{
			{ID: trainingForward, Phase: trainingprogram.PhaseForward},
			{ID: trainingBackward, Phase: trainingprogram.PhaseBackward},
			{ID: trainingMuon, Phase: trainingprogram.PhaseOptimize},
		},
		Parameters: []trainingprogram.ParameterSpec{
			{Name: "decoder.final_norm", Rows: d, Cols: 1, Trainable: true},
			{Name: "decoder.final_cross.raw_gate", Rows: 1, Cols: 1, Trainable: true},
			{Name: "decoder.final_cross.output", Rows: d, Cols: width, Trainable: true},
			{Name: "decoder.final_cross.input_norm", Rows: d, Cols: 1, Trainable: true},
			{Name: "decoder.final_cross.q_norm", Rows: model.Dims.HeadDim, Cols: 1, Trainable: true},
			{Name: "decoder.final_cross.k_norm", Rows: model.Dims.HeadDim, Cols: 1, Trainable: true},
			{Name: "decoder.final_cross.q", Rows: width, Cols: d, Trainable: true},
			{Name: "decoder.final_cross.k", Rows: kvWidth, Cols: d, Trainable: true},
			{Name: "decoder.final_cross.v", Rows: kvWidth, Cols: d, Trainable: true},
		},
		Optimizer: plan,
	})
	if err != nil {
		return nil, err
	}
	if baseLR <= 0 {
		baseLR = optimizer.DeriveBaseLR(layout.count)
	}
	weights := make([]float32, layout.count)
	copy(weights[layout.finalNorm.start:layout.finalNorm.end], model.decFinalNorm)
	block := &model.decoderCross[len(model.decoderCross)-1]
	weights[layout.rawGate.start] = block.rawGate
	for index, word := range block.o {
		weights[layout.output.start+index] = dtype.BF16ToFloat32(word)
	}
	copy(weights[layout.inputNorm.start:layout.inputNorm.end], block.inNorm)
	copy(weights[layout.qNorm.start:layout.qNorm.end], block.qNorm)
	copy(weights[layout.kNorm.start:layout.kNorm.end], block.kNorm)
	copyBF16ToFloat32(weights[layout.qProjection.start:layout.qProjection.end], block.q)
	copyBF16ToFloat32(weights[layout.kProjection.start:layout.kProjection.end], block.k)
	copyBF16ToFloat32(weights[layout.vProjection.start:layout.vProjection.end], block.v)
	gradients := make([]float32, layout.count)
	muon, err := newTrainingOptimizer(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: baseLR, Momentum: momentum, Steps: steps, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, err
	}
	trainer := &Trainer{
		model: model, layout: layout, program: program, weights: weights, gradients: gradients, optimizer: muon,
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
	if len(state.trace.finalCrossProjected) != len(dHidden) {
		return errors.New("final cross-attention trace differs")
	}
	var gateGradient float64
	for index, gradient := range dHidden {
		gateGradient += float64(gradient) * float64(state.trace.finalCrossProjected[index])
	}
	block := &t.model.decoderCross[len(t.model.decoderCross)-1]
	gate := block.gate
	state.gradient[t.layout.rawGate.start] = float32(gateGradient) * gate * (1 - gate)
	if len(state.trace.finalCrossAttention) != rows*t.model.Dims.Heads*t.model.Dims.HeadDim {
		return errors.New("final cross-attention core trace differs")
	}
	dProjected := make([]float32, len(dHidden))
	for index, gradient := range dHidden {
		dProjected[index] = gate * gradient
	}
	state.projectionGradient = state.gradient[t.layout.output.start:t.layout.output.end]
	hostmath.LinearBackward(
		nil, state.projectionGradient, nil, state.trace.finalCrossAttention, nil, dProjected,
		rows, t.model.Dims.Heads*t.model.Dims.HeadDim, d, false,
	)
	state.attentionQGradient = make([]float32, len(state.trace.finalCrossQ))
	state.attentionKGradient = make([]float32, len(state.trace.finalCrossK))
	state.attentionVGradient = make([]float32, len(state.trace.finalCrossV))
	dAttention := make([]float32, len(state.trace.finalCrossAttention))
	hostmath.LinearBF16BackwardInput(
		dAttention, dProjected, block.o,
		rows, t.model.Dims.Heads*t.model.Dims.HeadDim, d,
	)
	hostmath.MaskedBidirectionalAttentionBackward(
		state.attentionQGradient, state.attentionKGradient, state.attentionVGradient,
		state.trace.finalCrossQ, state.trace.finalCrossK, state.trace.finalCrossV, dAttention,
		rows, len(state.trace.finalCrossK)/(t.model.Dims.KVHeads*t.model.Dims.HeadDim),
		t.model.Dims.Heads, t.model.Dims.KVHeads, t.model.Dims.HeadDim, nil,
	)
	return t.backwardFinalCrossProjections(state, block)
}

func (t *Trainer) backwardFinalCrossProjections(state *trainingStep, block *attnBlock) error {
	dims := t.model.Dims
	rows := len(state.pair.Targets)
	memRows := len(state.trace.finalCrossMemory) / dims.DModel
	qWidth := dims.Heads * dims.HeadDim
	kvWidth := dims.KVHeads * dims.HeadDim
	if len(state.trace.finalCrossInput) != rows*dims.DModel ||
		len(state.trace.finalCrossNormed) != rows*dims.DModel ||
		len(state.trace.finalCrossQRaw) != rows*qWidth ||
		memRows == 0 || len(state.trace.finalCrossKRaw) != memRows*kvWidth {
		return errors.New("final cross-attention projection trace differs")
	}

	dQNorm := make([]float32, len(state.attentionQGradient))
	for index, value := range state.attentionQGradient {
		dQNorm[index] = value * t.model.scoreScale
	}
	dQRaw := make([]float32, len(dQNorm))
	hostmath.RMSNormBackward(
		dQRaw, state.gradient[t.layout.qNorm.start:t.layout.qNorm.end],
		state.trace.finalCrossQRaw, block.qNorm, dQNorm,
		rows*dims.Heads, dims.HeadDim, dims.RMSEps, false,
	)
	state.qProjectionGradient = state.gradient[t.layout.qProjection.start:t.layout.qProjection.end]
	hostmath.LinearBackward(
		nil, state.qProjectionGradient, nil, state.trace.finalCrossNormed, nil, dQRaw,
		rows, dims.DModel, qWidth, false,
	)
	dCrossNormed := make([]float32, len(state.trace.finalCrossNormed))
	hostmath.LinearBF16BackwardInput(dCrossNormed, dQRaw, block.q, rows, dims.DModel, qWidth)
	dCrossInput := make([]float32, len(dCrossNormed))
	hostmath.RMSNormBackward(
		dCrossInput, state.gradient[t.layout.inputNorm.start:t.layout.inputNorm.end],
		state.trace.finalCrossInput, block.inNorm, dCrossNormed,
		rows, dims.DModel, dims.RMSEps, false,
	)

	dKRaw := make([]float32, len(state.attentionKGradient))
	hostmath.RMSNormBackward(
		dKRaw, state.gradient[t.layout.kNorm.start:t.layout.kNorm.end],
		state.trace.finalCrossKRaw, block.kNorm, state.attentionKGradient,
		memRows*dims.KVHeads, dims.HeadDim, dims.RMSEps, false,
	)
	state.kProjectionGradient = state.gradient[t.layout.kProjection.start:t.layout.kProjection.end]
	state.vProjectionGradient = state.gradient[t.layout.vProjection.start:t.layout.vProjection.end]
	hostmath.LinearBackward(
		nil, state.kProjectionGradient, nil, state.trace.finalCrossMemory, nil, dKRaw,
		memRows, dims.DModel, kvWidth, false,
	)
	hostmath.LinearBackward(
		nil, state.vProjectionGradient, nil, state.trace.finalCrossMemory, nil, state.attentionVGradient,
		memRows, dims.DModel, kvWidth, false,
	)
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
	copy(t.model.decFinalNorm, t.weights[t.layout.finalNorm.start:t.layout.finalNorm.end])
	block := &t.model.decoderCross[len(t.model.decoderCross)-1]
	block.rawGate = t.weights[t.layout.rawGate.start]
	block.gate = sigmoid(block.rawGate)
	copyFloat32ToBF16(block.o, t.weights[t.layout.output.start:t.layout.output.end])
	copy(block.inNorm, t.weights[t.layout.inputNorm.start:t.layout.inputNorm.end])
	copy(block.qNorm, t.weights[t.layout.qNorm.start:t.layout.qNorm.end])
	copy(block.kNorm, t.weights[t.layout.kNorm.start:t.layout.kNorm.end])
	copyFloat32ToBF16(block.q, t.weights[t.layout.qProjection.start:t.layout.qProjection.end])
	copyFloat32ToBF16(block.k, t.weights[t.layout.kProjection.start:t.layout.kProjection.end])
	copyFloat32ToBF16(block.v, t.weights[t.layout.vProjection.start:t.layout.vProjection.end])
	return nil
}

func copyFloat32ToBF16(dst []uint16, src []float32) {
	for index, value := range src {
		dst[index] = dtype.Float32ToBF16(value)
	}
}
