// Minimal trainable graph for the forecast capability: the full-model VJP
// composed from the golden-verified component backwards (head block, decoder
// stack, patch embed) over the flat host Muon optimizer. The trainer packs
// every forward-path tensor into the optimizer's flat layout and re-points
// the model's named weights at the packed storage, so forward, backward and
// optimizer all operate on one copy. Trainable geometry derives from the
// artifact's own tensor shapes; nothing here asserts a model constant.
//
// The gradient map is pre-bound to the packed layout and audited after every
// backward: a gradient slot allocated outside the pack means a parameter
// would silently not train, and the step refuses instead.
package seriesforecast

import (
	"fmt"
	"math"
	"sort"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
)

// residualBlockTensors: the tensor suffixes a residual block may carry;
// biases are optional (the output heads ship without them).
var residualBlockTensors = []string{
	".hidden_layer.weight", ".hidden_layer.bias",
	".output_layer.weight", ".output_layer.bias",
	".residual_layer.weight", ".residual_layer.bias",
}

// layerTensors: the tensors every stacked_xf layer carries (enforced by
// layerWeights at load).
var layerTensors = []string{
	".pre_attn_ln.scale", ".post_attn_ln.scale", ".pre_ff_ln.scale", ".post_ff_ln.scale",
	".attn.qkv_proj.weight", ".attn.out.weight",
	".attn.query_ln.scale", ".attn.key_ln.scale", ".attn.per_dim_scale.per_dim_scale",
	".ff0.weight", ".ff1.weight",
}

// TrainStepResult: measured facts of one observed training step.
type TrainStepResult struct {
	MSE          float64
	Quantile     float64
	Total        float64
	Step         int
	LearningRate float64
	GradientL2   float64
}

// trainer: trainable graph over the packed parameter layout, stepped by the
// best available Muon backend (the shared CUDA stepper on this host, the
// flat host optimizer elsewhere — the same parity-gated pair the seq2seq
// training cells ran through).
type Trainer struct {
	model     *Model
	names     []string
	weights   []float32
	gradients []float32
	grads     Grads
	stepper   optimizer.Stepper
	config    optimizer.Config
	step      int
	invFreq   []float64
}

// trainableNames returns the forward-path tensor names in packing order:
// tokenizer block, decoder layers in stack order, then the point head.
func (m *Model) trainableNames() []string {
	var names []string
	appendBlock := func(prefix string) {
		for _, suffix := range residualBlockTensors {
			if m.Weights[prefix+suffix] != nil {
				names = append(names, prefix+suffix)
			}
		}
	}
	appendBlock("tokenizer")
	for index := 0; index < m.Dims.Layers; index++ {
		prefix := fmt.Sprintf("stacked_xf.%d", index)
		for _, suffix := range layerTensors {
			names = append(names, prefix+suffix)
		}
	}
	appendBlock("output_projection_point")
	return names
}

// NewTrainer packs the trainable tensors, re-points the model's weights at
// the packed storage, and compiles the Muon plan. A non-positive base
// learning rate derives n_params^-1/2.
func NewTrainer(model *Model, config optimizer.Config) (*Trainer, error) {
	names := model.trainableNames()
	specs := make([]optimizer.GroupSpec, len(names))
	total := 0
	for index, name := range names {
		values := model.Weights[name]
		shape := model.Shapes[name]
		rows, cols := 0, 0
		switch len(shape) {
		case 1:
			rows, cols = 1, shape[0]
		case 2:
			rows, cols = shape[0], shape[1]
		default:
			return nil, fmt.Errorf("seriesforecast: trainable %q has rank-%d shape", name, len(shape))
		}
		if rows*cols != len(values) {
			return nil, fmt.Errorf("seriesforecast: trainable %q shape %v does not cover %d values", name, shape, len(values))
		}
		specs[index] = optimizer.GroupSpec{Name: name, Start: total, End: total + len(values), Rows: rows, Cols: cols}
		total += len(values)
	}
	plan, err := optimizer.CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	if config.BaseLearningRate <= 0 {
		config.BaseLearningRate = optimizer.DeriveBaseLR(total)
	}
	trainer := &Trainer{
		model:     model,
		names:     names,
		weights:   make([]float32, total),
		gradients: make([]float32, total),
		grads:     make(Grads, len(names)),
		invFreq:   hostmath.RopeInvFreq(model.Dims.RopeTheta, model.Dims.HeadDim),
	}
	trainer.config = config
	for _, spec := range specs {
		copy(trainer.weights[spec.Start:spec.End], model.Weights[spec.Name])
		model.Weights[spec.Name] = trainer.weights[spec.Start:spec.End:spec.End]
		trainer.grads[spec.Name] = trainer.gradients[spec.Start:spec.End:spec.End]
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

// trainForward: the retained forward trace one step needs for backward.
type trainForward struct {
	padded, masks []float32
	mu, sigma     []float64
	tokens        int
	layerInputs   [][]float32
	lastToken     []float32
	forecast      []float32
}

// forward runs the retained forward pass and denormalized forecast.
func (t *Trainer) forward(input []float32) (trainForward, error) {
	m := t.model
	p, d := m.Dims.PatchLen, m.Dims.Hidden
	padded, masks, err := padToPatches(input, make([]float32, len(input)), p)
	if err != nil {
		return trainForward{}, err
	}
	tokens := len(padded) / p
	mu := make([]float64, tokens)
	sigma := make([]float64, tokens)
	patchStats(padded, masks, p, mu, sigma)
	hidden := make([]float32, tokens*d)
	if err := m.patchEmbed(hidden, padded, masks, mu, sigma); err != nil {
		return trainForward{}, err
	}
	layerInputs := make([][]float32, m.Dims.Layers)
	for index := 0; index < m.Dims.Layers; index++ {
		layerInputs[index] = append([]float32(nil), hidden...)
		l, err := m.layerWeights(index)
		if err != nil {
			return trainForward{}, err
		}
		m.layerForward(hidden, l, t.invFreq, tokens)
	}
	last := tokens - 1
	lastToken := hidden[last*d : (last+1)*d]
	out := make([]float32, m.Dims.Horizon*m.Dims.Quantiles)
	if err := m.residualBlock(out, "output_projection_point", lastToken); err != nil {
		return trainForward{}, err
	}
	forecast := make([]float32, len(out))
	for k := range out {
		forecast[k] = float32(float64(out[k])*sigma[last] + mu[last])
	}
	return trainForward{
		padded: padded, masks: masks, mu: mu, sigma: sigma,
		tokens: tokens, layerInputs: layerInputs, lastToken: lastToken, forecast: forecast,
	}, nil
}

// loss scores the forecast over the OBSERVED horizon prefix: a real future
// window shorter than the model horizon contributes loss (and gradient) only
// where ground truth exists; the uncovered tail carries zero gradient.
func (t *Trainer) loss(forecast, target []float32) (mse, quantile, total float64, grad []float32, err error) {
	m := t.model
	covered := len(target)
	if covered == 0 || covered > m.Dims.Horizon {
		return 0, 0, 0, nil, fmt.Errorf("seriesforecast: target covers %d of horizon %d", covered, m.Dims.Horizon)
	}
	return ForecastLoss(forecast[:covered*m.Dims.Quantiles], target, m.Levels, covered, m.Dims.Quantiles)
}

// Loss evaluates the covered-horizon objective without training.
func (t *Trainer) Loss(input, target []float32) (float64, error) {
	forward, err := t.forward(input)
	if err != nil {
		return 0, err
	}
	_, _, total, _, err := t.loss(forward.forecast, target)
	return total, err
}

// Step runs one observed training step: retained forward, forecast loss,
// full-model backward, gradient-coverage audit, Muon update.
func (t *Trainer) Step(input, target []float32) (TrainStepResult, error) {
	m := t.model
	d := m.Dims.Hidden
	forward, err := t.forward(input)
	if err != nil {
		return TrainStepResult{}, err
	}
	tokens, layerInputs := forward.tokens, forward.layerInputs
	mu, sigma := forward.mu, forward.sigma
	padded, masks, lastToken := forward.padded, forward.masks, forward.lastToken
	last := tokens - 1
	mse, quantile, total, dCovered, err := t.loss(forward.forecast, target)
	if err != nil {
		return TrainStepResult{}, err
	}
	dForecast := make([]float32, len(forward.forecast))
	copy(dForecast, dCovered)

	// Denormalization backward: forecast = out*sigma + mu with mu/sigma pure
	// data statistics, so only the sigma scale reaches the head.
	dOut := make([]float32, len(dForecast))
	for k := range dForecast {
		dOut[k] = float32(float64(dForecast[k]) * sigma[last])
	}
	dLast, err := m.residualBlockBackward("output_projection_point", lastToken, dOut, t.grads)
	if err != nil {
		return TrainStepResult{}, err
	}
	dHidden := make([]float32, tokens*d)
	copy(dHidden[last*d:(last+1)*d], dLast)
	for index := m.Dims.Layers - 1; index >= 0; index-- {
		dHidden, err = m.layerBackward(index, layerInputs[index], dHidden, t.invFreq, tokens, t.grads)
		if err != nil {
			return TrainStepResult{}, err
		}
	}
	if _, err := m.patchEmbedBackward(padded, masks, mu, sigma, dHidden, t.grads); err != nil {
		return TrainStepResult{}, err
	}
	if err := t.auditGradientCoverage(); err != nil {
		return TrainStepResult{}, err
	}
	var gradientSquared float64
	for _, gradient := range t.gradients {
		gradientSquared += float64(gradient) * float64(gradient)
	}
	if err := t.stepper.Step(); err != nil {
		return TrainStepResult{}, err
	}
	t.step++
	return TrainStepResult{
		MSE: mse, Quantile: quantile, Total: total,
		Step: t.step, LearningRate: t.config.LearningRate(t.step), GradientL2: math.Sqrt(gradientSquared),
	}, nil
}

// auditGradientCoverage refuses the step when backward touched a gradient
// slot outside the packed layout — that parameter would silently not train.
func (t *Trainer) auditGradientCoverage() error {
	if len(t.grads) == len(t.names) {
		return nil
	}
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
	return fmt.Errorf("seriesforecast: backward gradients outside the trainable pack: %v", extras)
}
