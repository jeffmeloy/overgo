// Trainer: compiled class-target bootstrap training.
package oscillatorimage

import (
	"fmt"
	"math"
	"math/rand"

	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

const (
	bootstrapInitScaleVector = 0.3
	bootstrapInitScaleConv   = 0.06
	classTargetBase          = 0.1
	classTargetStride        = 0.2
	classLossDerivative      = 2
)

// NewBootstrapModel constructs deterministic bootstrap weights.
func NewBootstrapModel(cfg Config, seed int64) *Model {
	rng := rand.New(rand.NewSource(seed))
	draw := func(count int, scale float64) []float32 {
		values := make([]float32, count)
		for i := range values {
			values[i] = float32(rng.NormFloat64() * scale)
		}
		return values
	}
	m := &Model{
		Cfg: cfg, Namespace: cfg.ModelType, Slope: decoderNegativeSlope,
		Omega:     draw(cfg.N, bootstrapInitScaleVector),
		OmegaCond: draw(cfg.NCond, bootstrapInitScaleVector),
		K:         draw(cfg.N*cfg.N, bootstrapInitScaleVector),
		KCond:     draw(cfg.NCond*cfg.NCond, bootstrapInitScaleVector),
		Drive:     draw(cfg.NClasses*cfg.N*cfg.NCond, bootstrapInitScaleVector),
	}
	cin := cfg.InChannels
	for _, cout := range cfg.BlockChannels {
		m.Blocks = append(m.Blocks, DecoderBlock{
			W1: draw(cout*cin*convTaps, bootstrapInitScaleConv), B1: draw(cout, bootstrapInitScaleVector),
			W2: draw(cout*cout*convTaps, bootstrapInitScaleConv), B2: draw(cout, bootstrapInitScaleVector),
			Cout: cout,
		})
		cin = cout
	}
	m.ToOutW = draw(cfg.OutChannels*cin*convTaps, bootstrapInitScaleConv)
	m.ToOutB = draw(cfg.OutChannels, bootstrapInitScaleVector)
	return m
}

// TrainStepResult: measured facts of one observed training step.
type TrainStepResult struct {
	Loss         float64
	Step         int
	LearningRate float64
	GradientL2   float64
}

type Trainer struct {
	model     *Model
	pack      *optimizer.TensorPack
	grads     Grads
	stepper   optimizer.Stepper
	execution trainingprogram.Execution[trainingState]
	config    optimizer.Config
	step      int
	rng       *rand.Rand
	init      []float32
	dImage    []float32
}

type trainingState struct {
	trainer *Trainer
	trace   trainingTrace
	result  TrainStepResult
}

type trainTensor struct {
	name       string
	rows, cols int
	field      *[]float32
}

// trainableLayout declares field geometry and ownership.
func (m *Model) trainableLayout() []trainTensor {
	n, nc, classes := len(m.Omega), len(m.OmegaCond), m.Cfg.NClasses
	layout := []trainTensor{
		{m.tensorName("omega"), n, 1, &m.Omega},
		{m.tensorName("omega_cond"), nc, 1, &m.OmegaCond},
		{m.tensorName("k"), n, n, &m.K},
		{m.tensorName("k_cond"), nc, nc, &m.KCond},
		{m.tensorName("k_drive"), classes, n * nc, &m.Drive},
	}
	for index := range m.Blocks {
		block := &m.Blocks[index]
		cout := block.Cout
		layout = append(layout,
			trainTensor{m.blockTensorName(index, "w1"), cout, len(block.W1) / cout, &block.W1},
			trainTensor{m.blockTensorName(index, "b1"), cout, 1, &block.B1},
			trainTensor{m.blockTensorName(index, "w2"), cout, len(block.W2) / cout, &block.W2},
			trainTensor{m.blockTensorName(index, "b2"), cout, 1, &block.B2},
		)
	}
	outRows := len(m.ToOutB)
	return append(layout,
		trainTensor{m.tensorName("to_out.weight"), outRows, len(m.ToOutW) / outRows, &m.ToOutW},
		trainTensor{m.tensorName("to_out.bias"), outRows, 1, &m.ToOutB},
	)
}

// NewTrainer binds model fields to one compiled optimizer pack.
func NewTrainer(model *Model, config optimizer.Config, seed int64) (*Trainer, error) {
	if model == nil {
		return nil, fmt.Errorf("oscillatorimage: model is required")
	}
	layout := model.trainableLayout()
	tensors := make(map[string][]float32, len(layout))
	shapes := make(map[string][2]int, len(layout))
	for _, tensor := range layout {
		values := *tensor.field
		if tensor.rows*tensor.cols != len(values) {
			return nil, fmt.Errorf("oscillatorimage: trainable %q geometry %dx%d does not cover %d values", tensor.name, tensor.rows, tensor.cols, len(values))
		}
		tensors[tensor.name] = values
		shapes[tensor.name] = [2]int{tensor.rows, tensor.cols}
	}
	pack, err := optimizer.NewTensorPack(tensors, optimizer.MatrixGeometry(shapes))
	if err != nil {
		return nil, err
	}
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveLatentL2, nil, pack.Plan())
	if err != nil {
		return nil, err
	}
	trainer := &Trainer{model: model, pack: pack, config: config, rng: rand.New(rand.NewSource(seed))}
	trainer.execution, err = trainingprogram.BindObjective(program, (*trainingState).forward, (*trainingState).backward, (*trainingState).optimize)
	if err != nil {
		return nil, err
	}
	trainer.stepper, err = pack.NewStepper(config)
	if err != nil {
		return nil, err
	}
	trainer.grads = pack.BindMapViews(tensors)
	for _, tensor := range layout {
		*tensor.field = tensors[tensor.name]
	}
	return trainer, nil
}

// ParameterCount reports the packed trainable parameter total.
func (t *Trainer) ParameterCount() int { return t.pack.ParameterCount() }

// TrainableParameterCount returns optimizer extent without allocation.
func (m *Model) TrainableParameterCount() (count int) {
	if m == nil {
		return count
	}
	for _, tensor := range m.trainableLayout() {
		count += len(*tensor.field)
	}
	return count
}

// Close releases the stepper's backend.
func (t *Trainer) Close() error { return t.stepper.Close() }

// classTargetLoss returns loss and optional image VJP.
func (m *Model) classTargetLoss(image []float32, dImage []float32) float64 {
	dim := m.Cfg.OutChannels * m.Cfg.OutH() * m.Cfg.OutW()
	var loss float64
	for class := 0; class < m.Cfg.NClasses; class++ {
		target := classTargetBase + classTargetStride*float64(class)
		for i := 0; i < dim; i++ {
			index := class*dim + i
			delta := float64(image[index]) - target
			loss += delta * delta
			if dImage != nil {
				dImage[index] = float32(classLossDerivative * delta)
			}
		}
	}
	return loss
}

// sampleInit fills uniform class phases.
func (t *Trainer) sampleInit(dst []float32) {
	for i := range dst {
		dst[i] = float32((t.rng.Float64()*2 - 1) * math.Pi)
	}
}

// Loss evaluates a fixed-seed objective.
func (t *Trainer) Loss(seed int64) float64 {
	m := t.model
	classes := m.Cfg.NClasses
	init := make([]float32, classes*(len(m.Omega)+len(m.OmegaCond)))
	rng := rand.New(rand.NewSource(seed))
	for i := range init {
		init[i] = float32((rng.Float64()*2 - 1) * math.Pi)
	}
	image, _ := m.trainingForwardTrace(init, m.Drive, classes)
	return m.classTargetLoss(image, nil)
}

func (t *Trainer) Step() (TrainStepResult, error) {
	state := trainingState{trainer: t}
	if err := t.execution.Run(&state); err != nil {
		return TrainStepResult{}, err
	}
	return state.result, nil
}

func (state *trainingState) forward() error {
	t := state.trainer
	m := t.model
	classes := m.Cfg.NClasses
	initSize := classes * (len(m.Omega) + len(m.OmegaCond))
	if cap(t.init) < initSize {
		t.init = make([]float32, initSize)
	} else {
		t.init = t.init[:initSize]
	}
	t.sampleInit(t.init)
	image, trace := m.trainingForwardTrace(t.init, m.Drive, classes)
	state.trace = trace
	if cap(t.dImage) < len(image) {
		t.dImage = make([]float32, len(image))
	} else {
		t.dImage = t.dImage[:len(image)]
	}
	state.result.Loss = m.classTargetLoss(image, t.dImage)
	return nil
}

func (state *trainingState) backward() error {
	t := state.trainer
	t.model.backwardInto(state.trace, t.model.Drive, t.dImage, t.grads)
	if len(t.grads) != t.pack.Plan().GroupCount() {
		return fmt.Errorf("oscillatorimage: backward gradients outside the trainable pack")
	}
	var gradientSquared float64
	for _, gradients := range t.grads {
		for _, gradient := range gradients {
			gradientSquared += float64(gradient) * float64(gradient)
		}
	}
	state.result.GradientL2 = math.Sqrt(gradientSquared)
	return nil
}

func (state *trainingState) optimize() error {
	t := state.trainer
	if err := t.stepper.Step(); err != nil {
		return err
	}
	t.step++
	state.result.Step = t.step
	state.result.LearningRate = t.config.LearningRate(t.step)
	return nil
}
