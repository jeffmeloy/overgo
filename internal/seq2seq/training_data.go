package seq2seq

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
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
	finalNormForward  = "seq2seq-forward"
	finalNormBackward = "final-norm-backward"
	finalNormMuon     = "muon"
)

// FinalNormTrainer is the first bounded Needle fine-tuning rung.
type FinalNormTrainer struct {
	model     *Model
	program   trainingprogram.TrainingProgram
	execution trainingprogram.Execution[finalNormStep]
	weights   []float32
	gradients []float32
	optimizer *optimizer.Optimizer
}

type finalNormStep struct {
	pair                   TrainingPair
	hidden, normed, logits []float32
	gradient               []float32
	loss                   float64
}

// NewFinalNormTrainer binds forward, backward, and Muon to compiled authority.
func NewFinalNormTrainer(model *Model, steps int, baseLR, momentum float64) (*FinalNormTrainer, error) {
	if model == nil || steps <= 0 {
		return nil, errors.New("seq2seq: invalid final-norm trainer")
	}
	d := model.Dims.DModel
	plan, err := optimizer.CompilePlan(d, []optimizer.GroupSpec{{
		Name: "decoder.final_norm", Start: 0, End: d, Rows: d, Cols: 1,
	}})
	if err != nil {
		return nil, err
	}
	program, err := trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Operators: []trainingprogram.OperatorSpec{
			{ID: finalNormForward, Phase: trainingprogram.PhaseForward},
			{ID: finalNormBackward, Phase: trainingprogram.PhaseBackward},
			{ID: finalNormMuon, Phase: trainingprogram.PhaseOptimize},
		},
		Parameters: []trainingprogram.ParameterSpec{{Name: "decoder.final_norm", Rows: d, Cols: 1, Trainable: true}},
		Optimizer:  plan,
	})
	if err != nil {
		return nil, err
	}
	if baseLR <= 0 {
		baseLR = optimizer.DeriveBaseLR(d)
	}
	weights := slices.Clone(model.decFinalNorm)
	gradients := make([]float32, d)
	muon, err := optimizer.New(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: baseLR, Momentum: momentum, Steps: steps, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, err
	}
	trainer := &FinalNormTrainer{
		model: model, program: program, weights: weights, gradients: gradients, optimizer: muon,
	}
	execution, err := trainingprogram.Bind(program, []trainingprogram.Binding[finalNormStep]{
		{Operator: finalNormForward, Execute: trainer.forward},
		{Operator: finalNormBackward, Execute: trainer.backward},
		{Operator: finalNormMuon, Execute: trainer.optimize},
	})
	if err != nil {
		return nil, err
	}
	trainer.execution = execution
	return trainer, nil
}

func (t *FinalNormTrainer) Program() trainingprogram.TrainingProgram { return t.program }

// Step executes the compiled forward/backward/optimize sequence once.
func (t *FinalNormTrainer) Step(pair TrainingPair) (float64, error) {
	if t == nil || t.optimizer == nil {
		return 0, errors.New("seq2seq: final-norm trainer unavailable")
	}
	state := finalNormStep{pair: pair}
	if err := t.execution.RunPhases(&state,
		trainingprogram.PhaseForward, trainingprogram.PhaseBackward, trainingprogram.PhaseOptimize,
	); err != nil {
		return 0, err
	}
	return state.loss, nil
}

func (t *FinalNormTrainer) forward(state *finalNormStep) error {
	if len(state.pair.Source) == 0 || len(state.pair.DecoderInput) == 0 || len(state.pair.DecoderInput) != len(state.pair.Targets) {
		return errors.New("invalid training pair")
	}
	memory, err := t.model.Encode(state.pair.Source)
	if err != nil {
		return err
	}
	state.hidden, err = t.model.decodeHiddenFull(memory, len(state.pair.Source), state.pair.DecoderInput)
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

func (t *FinalNormTrainer) backward(state *finalNormStep) error {
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
	state.gradient = make([]float32, d)
	dHidden := make([]float32, len(state.hidden))
	hostmath.RMSNormBackward(dHidden, state.gradient, state.hidden, t.model.decFinalNorm, dNormed, rows, d, t.model.Dims.RMSEps, false)
	return nil
}

func (t *FinalNormTrainer) optimize(state *finalNormStep) error {
	if len(state.gradient) != len(t.gradients) {
		return errors.New("final-norm gradient differs from parameter plan")
	}
	copy(t.gradients, state.gradient)
	t.optimizer.Step()
	copy(t.model.decFinalNorm, t.weights)
	return nil
}
