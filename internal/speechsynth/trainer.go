package speechsynth

import (
	"errors"

	"overgo/internal/checked"
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
	example TrainingExample
	loss    float64
}

// JointTrainer runs the compiled backbone+flow program through shared Muon.
// It is single-owner: steps, restoration and model inference must not overlap.
type JointTrainer struct {
	model           *Model
	pack            *optimizer.TensorPack
	stepper         optimizer.Stepper
	host            *optimizer.Optimizer
	config          optimizer.Config
	gradients       Grads
	weights         map[string][]float32
	checkpointReady bool
	program         trainingprogram.TrainingProgram
	execution       trainingprogram.Execution[jointTrainingState]
}

// NewJointTrainer selects host Muon explicitly when host is true; otherwise it
// preserves the platform stepper. The declared base rate is scaled for the
// joint parameter surface using DerivedJointLR.
func NewJointTrainer(model *Model, config optimizer.Config, host bool) (*JointTrainer, error) {
	if model == nil || config.Validate() != nil {
		return nil, errors.New("speechsynth: invalid joint trainer")
	}
	tensors, shapes := model.TrainedTensors(true)
	pack, err := optimizer.NewTensorPack(tensors, optimizer.MatrixGeometry(shapes))
	if err != nil {
		return nil, err
	}
	config.BaseLearningRate, _, _ = model.DerivedJointLR(config.BaseLearningRate)
	plan := pack.Plan()
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveLatentSequence, nil, plan)
	if err != nil {
		return nil, err
	}
	trainer := &JointTrainer{model: model, pack: pack, config: config, gradients: Grads{}, weights: tensors, program: program, checkpointReady: true}
	if host {
		trainer.host, err = pack.NewOptimizer(config)
	} else {
		trainer.stepper, err = pack.NewStepper(config)
	}
	if err != nil {
		return nil, err
	}
	execution, err := trainingprogram.BindObjective(program, trainer.forward, trainer.backward, trainer.optimize)
	if err != nil {
		_ = trainer.Close()
		return nil, err
	}
	trainer.execution = execution
	return trainer, nil
}

func (t *JointTrainer) Program() trainingprogram.TrainingProgram { return t.program }

func (t *JointTrainer) Step(example TrainingExample) (float64, error) {
	if t == nil || t.pack == nil || !t.checkpointReady {
		return 0, errors.New("speechsynth: joint trainer unavailable")
	}
	state := jointTrainingState{example: example}
	if err := t.execution.Run(&state); err != nil {
		return 0, err
	}
	t.checkpointReady = true
	return state.loss, nil
}

func (t *JointTrainer) Close() error {
	if t == nil || t.pack == nil {
		return nil
	}
	var err error
	if t.stepper != nil {
		err = t.stepper.Close()
	}
	t.pack, t.host, t.stepper, t.gradients = nil, nil, nil, nil
	t.weights = nil
	t.model = nil
	t.execution = trainingprogram.Execution[jointTrainingState]{}
	t.checkpointReady = false
	return err
}

// OptimizerSnapshot returns shared CPU Muon state at a completed step boundary.
// Device state is not converted to host precision or represented as resumable.
func (t *JointTrainer) OptimizerSnapshot() (optimizer.State, error) {
	if t == nil || t.host == nil || !t.checkpointReady || !t.finiteWeights() {
		return optimizer.State{}, errors.New("speechsynth: CPU checkpoint boundary unavailable")
	}
	return t.host.Snapshot(), nil
}

// Restore validates CPU state and every named weight before replacing either.
// Universal checkpoint, data-stream and artifact binding remain caller-owned.
func (t *JointTrainer) Restore(weights map[string][]float32, state optimizer.State) error {
	if t == nil || t.host == nil || state.Config != t.config {
		return errors.New("speechsynth: CPU restore configuration differs")
	}
	if err := optimizer.ValidateState(state, t.pack.Plan().Identity(), t.pack.ParameterCount()); err != nil {
		return err
	}
	if err := t.pack.RestoreWeights(weights); err != nil {
		return err
	}
	// Restore cannot fail after the same plan, shape and configuration checks;
	// the trainer contract excludes concurrent mutation of the supplied state.
	if err := t.host.Restore(state); err != nil {
		return err
	}
	t.checkpointReady = true
	return nil
}

func (t *JointTrainer) forward(state *jointTrainingState) error {
	example := state.example
	if checked.Empty(example.TextIDs) || !checked.PositiveInts(example.Frames) {
		return errors.New("speechsynth: invalid training example")
	}
	if err := checked.Length(example.Latents, example.Frames, t.model.Dims.LatentDim); err != nil {
		return errors.New("speechsynth: invalid training example")
	}
	for _, token := range example.TextIDs {
		if !checked.ValidIndex(token, t.model.Dims.TextVocab) {
			return errors.New("speechsynth: training token outside vocabulary")
		}
	}
	for _, value := range example.Latents {
		if !checked.Finite32(value) {
			return errors.New("speechsynth: non-finite training target")
		}
	}
	return nil
}

func (t *JointTrainer) backward(state *jointTrainingState) error {
	for _, values := range t.gradients {
		clear(values)
	}
	loss, err := t.model.LossAndGrads(
		state.example.TextIDs, state.example.Latents, state.example.Frames, state.example.Seed, t.gradients,
	)
	state.loss = loss
	if err == nil && !checked.Finite64(loss) {
		return errors.New("speechsynth: non-finite training loss")
	}
	return err
}

func (t *JointTrainer) optimize(state *jointTrainingState) error {
	for _, values := range t.gradients {
		for _, value := range values {
			if !checked.Finite32(value) {
				return errors.New("speechsynth: non-finite training gradient")
			}
		}
	}
	if err := t.pack.GatherGradients(t.gradients); err != nil {
		return err
	}
	t.checkpointReady = false
	if t.host != nil {
		t.host.Step()
	} else {
		if err := t.stepper.Step(); err != nil {
			return err
		}
	}
	t.pack.Scatter()
	if !t.finiteWeights() {
		return errors.New("speechsynth: non-finite update; restore before further training")
	}
	return nil
}

func (t *JointTrainer) finiteWeights() bool {
	for _, values := range t.weights {
		for _, value := range values {
			if !checked.Finite32(value) {
				return false
			}
		}
	}
	return true
}
