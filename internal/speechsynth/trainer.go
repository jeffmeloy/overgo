package speechsynth

import (
	"errors"

	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// TrainingExample binds tokenized text to normalized codec latents.
type TrainingExample struct {
	TextIDs []int
	Latents []float32
	Frames  int
	Seed    int64
}

type jointTrainingState struct {
	example   TrainingExample
	gradients Grads
	loss      float64
}

// JointTrainer runs the compiled backbone+flow program through shared Muon.
type JointTrainer struct {
	model     *Model
	pack      *optimizer.TensorPack
	stepper   optimizer.Stepper
	program   trainingprogram.TrainingProgram
	execution trainingprogram.Execution[jointTrainingState]
}

func NewJointTrainer(model *Model, steps int, baseLR, momentum float64) (*JointTrainer, error) {
	if model == nil || steps <= 0 || baseLR <= 0 {
		return nil, errors.New("speechsynth: invalid joint trainer")
	}
	tensors, shapes := model.TrainedTensors(true)
	pack, err := optimizer.NewTensorPack(tensors, optimizer.MatrixGeometry(shapes))
	if err != nil {
		return nil, err
	}
	learningRate, _, _ := model.DerivedJointLR(baseLR)
	stepper, err := pack.NewStepper(optimizer.Config{
		BaseLearningRate: learningRate, Momentum: momentum, Steps: steps, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, err
	}
	plan := pack.Plan()
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveLatentSequence, nil, plan)
	if err != nil {
		_ = stepper.Close()
		return nil, err
	}
	trainer := &JointTrainer{model: model, pack: pack, stepper: stepper, program: program}
	execution, err := trainingprogram.Bind(program, []trainingprogram.Binding[jointTrainingState]{
		{Operator: trainingprogram.ObjectiveOperatorForward, Execute: trainer.forward},
		{Operator: trainingprogram.ObjectiveOperatorBackward, Execute: trainer.backward},
		{Operator: trainingprogram.ObjectiveOperatorMuon, Execute: trainer.optimize},
	})
	if err != nil {
		_ = stepper.Close()
		return nil, err
	}
	trainer.execution = execution
	return trainer, nil
}

func (t *JointTrainer) Program() trainingprogram.TrainingProgram { return t.program }

func (t *JointTrainer) Step(example TrainingExample) (float64, error) {
	if t == nil || t.stepper == nil {
		return 0, errors.New("speechsynth: joint trainer unavailable")
	}
	state := jointTrainingState{example: example}
	if err := t.execution.RunPhases(&state,
		trainingprogram.PhaseForward, trainingprogram.PhaseBackward, trainingprogram.PhaseOptimize,
	); err != nil {
		return 0, err
	}
	return state.loss, nil
}

func (t *JointTrainer) Close() error {
	if t == nil || t.stepper == nil {
		return nil
	}
	return t.stepper.Close()
}

func (t *JointTrainer) forward(state *jointTrainingState) error {
	example := state.example
	if len(example.TextIDs) == 0 || example.Frames <= 0 || len(example.Latents) != example.Frames*t.model.Dims.LatentDim {
		return errors.New("speechsynth: invalid training example")
	}
	for _, token := range example.TextIDs {
		if token < 0 || token >= t.model.Dims.TextVocab {
			return errors.New("speechsynth: training token outside vocabulary")
		}
	}
	return nil
}

func (t *JointTrainer) backward(state *jointTrainingState) error {
	state.gradients = Grads{}
	loss, err := t.model.LossAndGrads(
		state.example.TextIDs, state.example.Latents, state.example.Frames, state.example.Seed, state.gradients,
	)
	state.loss = loss
	return err
}

func (t *JointTrainer) optimize(state *jointTrainingState) error {
	if err := t.pack.GatherGradients(state.gradients); err != nil {
		return err
	}
	if err := t.stepper.Step(); err != nil {
		return err
	}
	t.pack.Scatter()
	return nil
}
