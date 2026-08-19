// Trainer for the conditional-oscillator image capability: the recovered
// reference bootstrap semantics (adaptive_new extmodel oscillator training,
// deleted upstream in cleanup 581a1cef3 and ported forward here) over the
// package's finite-difference-verified BPTT backward. The objective is the
// reference's class-separation bootstrap: each class's whole image trains
// toward the constant 0.1 + 0.2*class under fresh uniform phase inits per
// step; Muon runs at the reference's lr 0.02, momentum 0.95 through the
// shared stepper. Geometry mirrors the reference layout tensor for tensor.
//
// The gradient map is pre-bound to the packed layout and audited after
// every backward: a slot outside the pack refuses the step.
package oscillatorimage

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"overgo/internal/optimizer"
)

// Reference bootstrap init scales (adaptive_new shipped policy, measured
// and defended in its init-scale A/B before deletion).
const (
	bootstrapInitScaleVector = 0.3
	bootstrapInitScaleConv   = 0.06
)

// NewBootstrapModel constructs a scratch model for a config under the
// reference's shipped init policy: normal draws at scale 0.3 for dynamics
// and bias tensors, 0.06 for convolution kernels.
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

// Trainer: packed trainable graph over the shared Muon stepper.
type Trainer struct {
	model     *Model
	names     []string
	weights   []float32
	gradients []float32
	grads     Grads
	stepper   optimizer.Stepper
	config    optimizer.Config
	step      int
	rng       *rand.Rand
}

type trainTensor struct {
	name       string
	rows, cols int
	field      *[]float32
}

// trainableLayout mirrors the reference bootstrap layout: dynamics tensors,
// decoder blocks in cascade order, then the output convolution.
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

// NewTrainer packs the trainable tensors, re-points the model's fields at
// the packed storage, and binds the reference training policy (lr 0.02,
// momentum 0.95) unless the config overrides it.
func NewTrainer(model *Model, config optimizer.Config, seed int64) (*Trainer, error) {
	if config.BaseLearningRate <= 0 {
		config.BaseLearningRate = 0.02
	}
	if config.Momentum <= 0 {
		config.Momentum = 0.95
	}
	layout := model.trainableLayout()
	specs := make([]optimizer.GroupSpec, len(layout))
	total := 0
	for index, tensor := range layout {
		values := *tensor.field
		if tensor.rows*tensor.cols != len(values) {
			return nil, fmt.Errorf("oscillatorimage: trainable %q geometry %dx%d does not cover %d values", tensor.name, tensor.rows, tensor.cols, len(values))
		}
		specs[index] = optimizer.GroupSpec{Name: tensor.name, Start: total, End: total + len(values), Rows: tensor.rows, Cols: tensor.cols}
		total += len(values)
	}
	plan, err := optimizer.CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	trainer := &Trainer{
		model: model, config: config,
		weights:   make([]float32, total),
		gradients: make([]float32, total),
		grads:     make(Grads, len(layout)),
		rng:       rand.New(rand.NewSource(seed)),
	}
	for index, tensor := range layout {
		spec := specs[index]
		copy(trainer.weights[spec.Start:spec.End], *tensor.field)
		*tensor.field = trainer.weights[spec.Start:spec.End:spec.End]
		trainer.grads[spec.Name] = trainer.gradients[spec.Start:spec.End:spec.End]
		trainer.names = append(trainer.names, spec.Name)
	}
	trainer.stepper, err = optimizer.NewStepper(trainer.weights, trainer.gradients, plan, config)
	if err != nil {
		return nil, err
	}
	return trainer, nil
}

// ParameterCount reports the packed trainable parameter total.
func (t *Trainer) ParameterCount() int { return len(t.weights) }

// Close releases the stepper's backend.
func (t *Trainer) Close() error { return t.stepper.Close() }

// classTargetLoss: the reference bootstrap objective — class c's image
// trains toward the constant 0.1 + 0.2*c; returns loss and dLoss/dImage.
func (m *Model) classTargetLoss(image []float32, dImage []float32) float64 {
	dim := m.Cfg.OutChannels * m.Cfg.OutH() * m.Cfg.OutW()
	var loss float64
	for class := 0; class < m.Cfg.NClasses; class++ {
		target := 0.1 + 0.2*float64(class)
		for i := 0; i < dim; i++ {
			index := class*dim + i
			delta := float64(image[index]) - target
			loss += delta * delta
			if dImage != nil {
				dImage[index] = float32(2 * delta)
			}
		}
	}
	return loss
}

// sampleInit fills a fresh uniform phase init for every class row.
func (t *Trainer) sampleInit(dst []float32) {
	for i := range dst {
		dst[i] = float32((t.rng.Float64()*2 - 1) * math.Pi)
	}
}

// Loss evaluates the bootstrap objective on a fixed seeded init without
// training (the before/after descent probe).
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

// Step runs one observed training step: fresh phase inits, BPTT backward,
// gradient-coverage audit, Muon update.
func (t *Trainer) Step() (TrainStepResult, error) {
	m := t.model
	classes := m.Cfg.NClasses
	init := make([]float32, classes*(len(m.Omega)+len(m.OmegaCond)))
	t.sampleInit(init)
	image, trace := m.trainingForwardTrace(init, m.Drive, classes)
	dImage := make([]float32, len(image))
	loss := m.classTargetLoss(image, dImage)
	m.backwardInto(trace, m.Drive, dImage, t.grads)
	if len(t.grads) != len(t.names) {
		packed := make(map[string]struct{}, len(t.names))
		for _, name := range t.names {
			packed[name] = struct{}{}
		}
		var extras []string
		for name := range t.grads {
			if _, ok := packed[name]; !ok {
				extras = append(extras, name)
			}
		}
		sort.Strings(extras)
		return TrainStepResult{}, fmt.Errorf("oscillatorimage: backward gradients outside the trainable pack: %v", extras)
	}
	var gradientSquared float64
	for _, gradient := range t.gradients {
		gradientSquared += float64(gradient) * float64(gradient)
	}
	if err := t.stepper.Step(); err != nil {
		return TrainStepResult{}, err
	}
	// The stepper consumes matrix-group gradients; clear the buffer whole so
	// the next backward accumulates from zero regardless of group geometry.
	clear(t.gradients)
	t.step++
	return TrainStepResult{
		Loss: loss, Step: t.step,
		LearningRate: t.config.LearningRate(t.step), GradientL2: math.Sqrt(gradientSquared),
	}, nil
}
