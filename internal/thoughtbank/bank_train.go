package thoughtbank

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// BankTrainingExample binds ordered hidden rows, memory, and target rows.
type BankTrainingExample struct {
	Input, Memory, Target []float32
	Rows, Slots           int
}

// BankStepResult reports the bound objective and compiled program.
type BankStepResult struct {
	Loss      float64
	ProgramID string
}

type bankStepState struct {
	example  BankTrainingExample
	output   []float32
	dOutput  []float32
	gradient *FastWeightBankReadGradients
	loss     float64
}

// FastWeightBankTrainer runs bank-read training through shared program and Muon owners.
type FastWeightBankTrainer struct {
	weights   *FastWeightBankWeights
	pack      *optimizer.TensorPack
	stepper   optimizer.Stepper
	execution trainingprogram.Execution[bankStepState]
	closed    bool
}

type bankStepperFactory func(*optimizer.TensorPack, optimizer.Config) (optimizer.Stepper, error)

// NewTrainer compiles one reusable bank training session.
func (weights *FastWeightBankWeights) NewTrainer(config optimizer.Config) (*FastWeightBankTrainer, error) {
	return newFastWeightBankTrainer(weights, config, func(pack *optimizer.TensorPack, config optimizer.Config) (optimizer.Stepper, error) {
		return pack.NewStepper(config)
	})
}

func newFastWeightBankTrainer(weights *FastWeightBankWeights, config optimizer.Config, factory bankStepperFactory) (*FastWeightBankTrainer, error) {
	if weights == nil || factory == nil {
		return nil, errors.New("thoughtbank training: incomplete binding")
	}
	if err := weights.Validate(); err != nil {
		return nil, err
	}
	tensors := map[string][]float32{
		"bank.fw_a": weights.FWA,
		"bank.fw_b": weights.FWB,
		"bank.fw_o": weights.FWO,
		"bank.norm": weights.NormWeight,
	}
	na := 1
	if weights.SwiGLU {
		na = 2
	}
	shapes := map[string][2]int{
		"bank.fw_a": {na * weights.Rank * weights.DModel, weights.MemDim},
		"bank.fw_b": {weights.DModel * weights.Rank, weights.MemDim},
		"bank.fw_o": {weights.DModel, weights.DModel},
		"bank.norm": {weights.DModel, 1},
	}
	pack, err := optimizer.NewTensorPack(tensors, optimizer.MatrixGeometry(shapes))
	if err != nil {
		return nil, err
	}
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveTokenPrediction, nil, pack.Plan())
	if err != nil {
		return nil, err
	}
	stepper, err := factory(pack, config)
	if err != nil {
		return nil, err
	}
	trainer := &FastWeightBankTrainer{weights: weights, pack: pack, stepper: stepper}
	execution, err := trainingprogram.BindObjective(program, trainer.forward, trainer.backward, trainer.optimize)
	if err != nil {
		_ = stepper.Close()
		return nil, err
	}
	trainer.execution = execution
	return trainer, nil
}

// Step executes one complete compiled update.
func (t *FastWeightBankTrainer) Step(example BankTrainingExample) (BankStepResult, error) {
	if t == nil || t.closed {
		return BankStepResult{}, errors.New("thoughtbank training: session unavailable")
	}
	state := bankStepState{example: example}
	if err := t.execution.Run(&state); err != nil {
		return BankStepResult{}, err
	}
	return BankStepResult{Loss: state.loss, ProgramID: t.execution.ProgramID().String()}, nil
}

func (t *FastWeightBankTrainer) forward(state *bankStepState) error {
	example := state.example
	d := t.weights.DModel
	if example.Rows <= 0 || example.Slots <= 0 || len(example.Input) != example.Rows*d ||
		len(example.Target) != example.Rows*d || len(example.Memory) != example.Slots*t.weights.MemDim {
		return errors.New("example shape differs from bank training contract")
	}
	output, err := FastWeightBankRead(example.Input, example.Memory, example.Rows, example.Slots, t.weights)
	if err != nil {
		return err
	}
	state.output = output
	state.dOutput = make([]float32, len(output))
	scale := 1.0 / float64(len(output))
	for index, value := range output {
		delta := float64(value) - float64(example.Target[index])
		state.loss += delta * delta * scale
		state.dOutput[index] = float32(2 * delta * scale)
	}
	if math.IsNaN(state.loss) || math.IsInf(state.loss, 0) {
		return errors.New("non-finite bank training loss")
	}
	return nil
}

func (t *FastWeightBankTrainer) backward(state *bankStepState) error {
	gradient, err := FastWeightBankReadBackward(
		state.example.Input, state.example.Memory, state.dOutput,
		state.example.Rows, state.example.Slots, t.weights,
	)
	if err != nil {
		return err
	}
	state.gradient = gradient
	return t.pack.GatherGradients(map[string][]float32{
		"bank.fw_a": gradient.DFWA,
		"bank.fw_b": gradient.DFWB,
		"bank.fw_o": gradient.DFWO,
		"bank.norm": gradient.DNormWeight,
	})
}

func (t *FastWeightBankTrainer) optimize(*bankStepState) error {
	if err := t.stepper.Step(); err != nil {
		return fmt.Errorf("shared Muon: %w", err)
	}
	t.pack.Scatter()
	return nil
}

// Close releases the selected optimizer backend.
func (t *FastWeightBankTrainer) Close() error {
	if t == nil || t.closed {
		return nil
	}
	t.closed = true
	return t.stepper.Close()
}
