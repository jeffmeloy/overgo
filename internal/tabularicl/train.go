package tabularicl

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

const (
	decoderInputWeight  = "icl_predictor.decoder.layers.0.weight"
	decoderInputBias    = "icl_predictor.decoder.layers.0.bias"
	decoderOutputWeight = "icl_predictor.decoder.layers.1.weight"
	decoderOutputBias   = "icl_predictor.decoder.layers.1.bias"
)

// TrainStepResult: one committed decoder update.
type TrainStepResult struct {
	Loss         float64
	Step         int
	LearningRate float64
	GradientL2   float64
}

// Trainer updates the decoder over frozen support-conditioned features.
type Trainer struct {
	head      *Head
	pack      *optimizer.TensorPack
	gradients map[string][]float32
	stepper   optimizer.Stepper
	config    optimizer.Config
	program   trainingprogram.TrainingProgram
	execution trainingprogram.Execution[decoderTrainingStep]
	step      int
}

// NewTrainer binds the decoder directly to optimizer storage.
func NewTrainer(head *Head, config optimizer.Config) (*Trainer, error) {
	if head == nil || head.Dims.NumCLS <= 0 || head.Dims.EmbedDim <= 0 || head.Dims.OutDim <= 0 ||
		len(head.decL0B) == 0 || len(head.decL1B) != head.Dims.OutDim {
		return nil, errors.New("tabularicl trainer: invalid decoder")
	}
	input := head.Dims.NumCLS * head.Dims.EmbedDim
	hidden := len(head.decL0B)
	shapes := map[string][2]int{
		decoderInputWeight:  {hidden, input},
		decoderInputBias:    {hidden, 1},
		decoderOutputWeight: {head.Dims.OutDim, hidden},
		decoderOutputBias:   {head.Dims.OutDim, 1},
	}
	tensors := map[string][]float32{
		decoderInputWeight: head.decL0W, decoderInputBias: head.decL0B,
		decoderOutputWeight: head.decL1W, decoderOutputBias: head.decL1B,
	}
	pack, err := optimizer.NewTensorPack(tensors, optimizer.MatrixGeometry(shapes))
	if err != nil {
		return nil, err
	}
	if config.BaseLearningRate <= 0 {
		config.BaseLearningRate = optimizer.DeriveBaseLR(pack.ParameterCount())
	}
	gradients := pack.BindMapViews(tensors)
	head.decL0W, head.decL0B = tensors[decoderInputWeight], tensors[decoderInputBias]
	head.decL1W, head.decL1B = tensors[decoderOutputWeight], tensors[decoderOutputBias]
	stepper, err := pack.NewStepper(config)
	if err != nil {
		return nil, err
	}
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveTablePrediction, nil, pack.Plan())
	if err != nil {
		_ = stepper.Close()
		return nil, err
	}
	trainer := &Trainer{head: head, pack: pack, gradients: gradients, stepper: stepper, config: config, program: program}
	trainer.execution, err = trainingprogram.BindObjective(program, trainer.forward, trainer.backward, func(state *decoderTrainingStep) error {
		if err := trainer.stepper.Step(); err != nil {
			return err
		}
		trainer.step++
		state.result.Step = trainer.step
		state.result.LearningRate = trainer.config.LearningRate(trainer.step)
		return nil
	})
	if err != nil {
		_ = stepper.Close()
		return nil, err
	}
	return trainer, nil
}

func (trainer *Trainer) Close() error { return trainer.stepper.Close() }

func (trainer *Trainer) ParameterCount() int { return trainer.pack.ParameterCount() }

type decoderTrainingStep struct {
	request                Request
	input, pre, activated  []float32
	output, outputGradient []float32
	result                 TrainStepResult
}

// Step runs one query-loss update; support rows remain conditioning only.
func (trainer *Trainer) Step(request Request) (TrainStepResult, error) {
	state := decoderTrainingStep{request: request}
	err := trainer.execution.Run(&state)
	return state.result, err
}

func (trainer *Trainer) forward(state *decoderTrainingStep) error {
	request := state.request
	if err := ValidateRequest(request); err != nil {
		return err
	}
	mask, err := requestCatMask(request.X, request.Y, request.Rows, request.Cols, request.TrainRows, request.CatCols)
	if err != nil {
		return err
	}
	head := trainer.head
	if head.Dims.IsClassifier != (request.Task == TaskClassification) {
		return errors.New("tabularicl trainer: task differs from decoder")
	}
	state.input = head.decoderInput(request.X, request.Y, request.Rows, request.Cols, request.TrainRows, mask)
	hidden := len(head.decL0B)
	state.pre = make([]float32, request.Rows*hidden)
	hostmath.Linear(state.pre, state.input, head.decL0W, request.Rows, head.Dims.NumCLS*head.Dims.EmbedDim, hidden)
	addBiasRows(state.pre, head.decL0B, request.Rows, hidden)
	state.activated = append([]float32(nil), state.pre...)
	hostmath.GELUTanhInPlace(state.activated)
	state.output = make([]float32, request.Rows*head.Dims.OutDim)
	hostmath.Linear(state.output, state.activated, head.decL1W, request.Rows, hidden, head.Dims.OutDim)
	addBiasRows(state.output, head.decL1B, request.Rows, head.Dims.OutDim)
	state.outputGradient = make([]float32, len(state.output))
	return nil
}

func (trainer *Trainer) backward(state *decoderTrainingStep) error {
	head, request := trainer.head, state.request
	queryRows := request.Rows - request.TrainRows
	start := request.TrainRows * head.Dims.OutDim
	queryOutput := state.output[start:]
	queryGradient := state.outputGradient[start:]
	if head.Dims.IsClassifier {
		targets := make([]int, queryRows)
		for index := range targets {
			value := request.Y[request.TrainRows+index]
			targets[index] = int(value)
			if float32(targets[index]) != value || targets[index] < 0 || targets[index] >= head.Dims.OutDim {
				return fmt.Errorf("tabularicl trainer: query row %d class %g outside decoder", index, value)
			}
		}
		state.result.Loss = hostmath.SoftmaxCrossEntropy(queryGradient, queryOutput, targets, queryRows, head.Dims.OutDim)
	} else {
		if head.Dims.OutDim != 1 {
			return errors.New("tabularicl trainer: regression decoder must emit one value")
		}
		inverse := 1 / float64(queryRows)
		for index := range queryRows {
			difference := float64(queryOutput[index] - request.Y[request.TrainRows+index])
			state.result.Loss += difference * difference * inverse
			queryGradient[index] = float32(2 * difference * inverse)
		}
	}
	hidden := len(head.decL0B)
	dActivated := make([]float32, len(state.activated))
	hostmath.LinearBackward(dActivated, trainer.gradients[decoderOutputWeight], trainer.gradients[decoderOutputBias],
		state.activated, head.decL1W, state.outputGradient, request.Rows, hidden, head.Dims.OutDim, false)
	hostmath.GELUTanhBackward(dActivated, state.pre, dActivated)
	hostmath.LinearBackward(nil, trainer.gradients[decoderInputWeight], trainer.gradients[decoderInputBias],
		state.input, nil, dActivated, request.Rows, head.Dims.NumCLS*head.Dims.EmbedDim, hidden, false)
	var squared float64
	for _, gradient := range trainer.gradients {
		for _, value := range gradient {
			squared += float64(value) * float64(value)
		}
	}
	state.result.GradientL2 = math.Sqrt(squared)
	if math.IsNaN(state.result.Loss) || math.IsInf(state.result.Loss, 0) || math.IsNaN(state.result.GradientL2) || math.IsInf(state.result.GradientL2, 0) {
		return errors.New("tabularicl trainer: non-finite loss or gradient")
	}
	return nil
}
