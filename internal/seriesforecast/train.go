package seriesforecast

import (
	"fmt"
	"math"
	"sort"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// residualBlockTensors: optional-bias residual layout.
var residualBlockTensors = []string{
	".hidden_layer.weight", ".hidden_layer.bias",
	".output_layer.weight", ".output_layer.bias",
	".residual_layer.weight", ".residual_layer.bias",
}

// layerTensors: validated decoder-layer layout.
var layerTensors = []string{
	".pre_attn_ln.scale", ".post_attn_ln.scale", ".pre_ff_ln.scale", ".post_ff_ln.scale",
	".attn.qkv_proj.weight", ".attn.out.weight",
	".attn.query_ln.scale", ".attn.key_ln.scale", ".attn.per_dim_scale.per_dim_scale",
	".ff0.weight", ".ff1.weight",
}

type TrainStepResult struct {
	MSE          float64
	Quantile     float64
	Total        float64
	Step         int
	LearningRate float64
	GradientL2   float64
}

type Trainer struct {
	model     *Model
	pack      *optimizer.TensorPack
	names     []string
	grads     Grads
	stepper   optimizer.Stepper
	program   trainingprogram.TrainingProgram
	execution trainingprogram.Execution[trainingStep]
	config    optimizer.Config
	step      int
	invFreq   []float64
}

// trainableNames: tokenizer, decoder stack, point head.
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

// NewTrainer binds model views to compiled training storage.
func NewTrainer(model *Model, config optimizer.Config) (*Trainer, error) {
	names := model.trainableNames()
	tensors := make(map[string][]float32, len(names))
	for _, name := range names {
		tensors[name] = model.Weights[name]
	}
	pack, err := optimizer.NewTensorPack(tensors, func(name string, length int) (int, int, error) {
		shape := model.Shapes[name]
		rows, cols := 0, 0
		switch len(shape) {
		case 1:
			rows, cols = 1, shape[0]
		case 2:
			rows, cols = shape[0], shape[1]
		default:
			return 0, 0, fmt.Errorf("trainable %q has rank-%d shape", name, len(shape))
		}
		if rows*cols != length {
			return 0, 0, fmt.Errorf("trainable %q shape %v does not cover %d values", name, shape, length)
		}
		return rows, cols, nil
	})
	if err != nil {
		return nil, err
	}
	if config.BaseLearningRate <= 0 {
		config.BaseLearningRate = trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(pack.ParameterCount())
	}
	trainer := &Trainer{
		model: model, pack: pack, names: names, config: config,
		invFreq: hostmath.RopeInvFreq(model.Dims.RopeTheta, model.Dims.HeadDim),
	}
	trainer.grads = pack.BindMapViews(model.Weights)
	trainer.stepper, err = pack.NewStepper(config)
	if err != nil {
		return nil, err
	}
	trainer.program, err = trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveForecast, nil, pack.Plan())
	if err != nil {
		_ = trainer.stepper.Close()
		return nil, err
	}
	trainer.execution, err = trainingprogram.BindObjective(
		trainer.program, trainer.forwardStep, trainer.backwardStep, func(state *trainingStep) error {
			if err := trainer.stepper.Step(); err != nil {
				return err
			}
			trainer.step++
			state.result.Step = trainer.step
			state.result.LearningRate = trainer.config.LearningRate(trainer.step)
			return nil
		},
	)
	if err != nil {
		_ = trainer.stepper.Close()
		return nil, err
	}
	return trainer, nil
}

func (t *Trainer) ParameterCount() int { return t.pack.ParameterCount() }

func (t *Trainer) Program() trainingprogram.TrainingProgram { return t.program }

func (t *Trainer) Close() error { return t.stepper.Close() }

type trainForward struct {
	padded, masks []float32
	mu, sigma     []float64
	tokens        int
	layerInputs   [][]float32
	lastToken     []float32
	forecast      []float32
}

type trainingStep struct {
	input, target []float32
	forward       trainForward
	result        TrainStepResult
}

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

// loss: observed horizon only; uncovered tail has zero gradient.
func (t *Trainer) loss(forecast, target []float32) (mse, quantile, total float64, grad []float32, err error) {
	m := t.model
	covered := len(target)
	if covered == 0 || covered > m.Dims.Horizon {
		return 0, 0, 0, nil, fmt.Errorf("seriesforecast: target covers %d of horizon %d", covered, m.Dims.Horizon)
	}
	return ForecastLoss(forecast[:covered*m.Dims.Quantiles], target, m.Levels, covered, m.Dims.Quantiles)
}

func (t *Trainer) Loss(input, target []float32) (float64, error) {
	forward, err := t.forward(input)
	if err != nil {
		return 0, err
	}
	_, _, total, _, err := t.loss(forward.forecast, target)
	return total, err
}

func (t *Trainer) Step(input, target []float32) (TrainStepResult, error) {
	state := trainingStep{input: input, target: target}
	err := t.execution.Run(&state)
	return state.result, err
}

func (t *Trainer) forwardStep(state *trainingStep) (err error) {
	state.forward, err = t.forward(state.input)
	return err
}

func (t *Trainer) backwardStep(state *trainingStep) error {
	m := t.model
	d := m.Dims.Hidden
	forward, target := state.forward, state.target
	tokens, layerInputs := forward.tokens, forward.layerInputs
	mu, sigma := forward.mu, forward.sigma
	padded, masks, lastToken := forward.padded, forward.masks, forward.lastToken
	last := tokens - 1
	mse, quantile, total, dCovered, err := t.loss(forward.forecast, target)
	if err != nil {
		return err
	}
	dForecast := make([]float32, len(forward.forecast))
	copy(dForecast, dCovered)

	// Data statistics are constants; only sigma reaches the head.
	dOut := make([]float32, len(dForecast))
	for k := range dForecast {
		dOut[k] = float32(float64(dForecast[k]) * sigma[last])
	}
	dLast, err := m.residualBlockBackward("output_projection_point", lastToken, dOut, t.grads)
	if err != nil {
		return err
	}
	dHidden := make([]float32, tokens*d)
	copy(dHidden[last*d:(last+1)*d], dLast)
	for index := m.Dims.Layers - 1; index >= 0; index-- {
		dHidden, err = m.layerBackward(index, layerInputs[index], dHidden, t.invFreq, tokens, t.grads)
		if err != nil {
			return err
		}
	}
	if _, err := m.patchEmbedBackward(padded, masks, mu, sigma, dHidden, t.grads); err != nil {
		return err
	}
	if err := t.auditGradientCoverage(); err != nil {
		return err
	}
	var gradientSquared float64
	for _, gradients := range t.grads {
		for _, gradient := range gradients {
			gradientSquared += float64(gradient) * float64(gradient)
		}
	}
	state.result = TrainStepResult{MSE: mse, Quantile: quantile, Total: total, GradientL2: math.Sqrt(gradientSquared)}
	return nil
}

func (t *Trainer) auditGradientCoverage() error {
	if len(t.grads) == len(t.names) {
		for _, name := range t.names {
			if len(t.grads[name]) != len(t.model.Weights[name]) {
				return fmt.Errorf("seriesforecast: backward gradient %q differs from trainable storage", name)
			}
		}
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
