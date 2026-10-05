// Package loratrain trains a LoRA adapter on a dense causal model. The base
// weights stay frozen: each adapted projection W is replaced in the model by
// W + scale·B·A of the current low-rank factors, the existing backward pass
// gives the gradient of the loss with respect to that effective weight, and the
// factors' gradients are projected from it. Our Muon optimizer steps only the
// factors.
package loratrain

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/densecausal"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/peft"
	"overgo/internal/processmeasure"
)

// The names of an adapter's two factors in the optimizer's tensor pack: the
// module path, then which factor.
const (
	downFactor = ".lora_A"
	upFactor   = ".lora_B"
	// weightSuffix is the tensor an adapted module's update lands in.
	weightSuffix = ".weight"
	// matrixRank is the number of extents of a projection weight, out by in, and
	// of each of an adapter's factors.
	matrixRank = 2
	// uniformSpan is the width of [-1, 1): a draw on [0, 1) times it, less one,
	// lands there.
	uniformSpan = 2
)

// projections are the seven linear projections of a dense decoder layer, the
// layers a LoRA adapter on every linear layer adapts, each by the name a target
// goes by and the weight the model names it under (a layer without one, as a
// mixture layer has no dense gate, up or down, is skipped).
var projections = []struct {
	target string
	weight func(densecausal.LayerTensorNames) string
}{
	{"q_proj", func(names densecausal.LayerTensorNames) string { return names.Q }},
	{"k_proj", func(names densecausal.LayerTensorNames) string { return names.K }},
	{"v_proj", func(names densecausal.LayerTensorNames) string { return names.V }},
	{"o_proj", func(names densecausal.LayerTensorNames) string { return names.O }},
	{"gate_proj", func(names densecausal.LayerTensorNames) string { return names.Gate }},
	{"up_proj", func(names densecausal.LayerTensorNames) string { return names.Up }},
	{"down_proj", func(names densecausal.LayerTensorNames) string { return names.Down }},
}

// Config names the adapter a fresh training starts.
type Config struct {
	// Rank and Alpha are every module's: its scale is Alpha over Rank.
	Rank  int
	Alpha float64
	// Targets are the projection names to adapt; empty adapts all seven.
	Targets []string
	// Seed makes the factors' initialization repeat.
	Seed uint64
	// BaseModel names the model the adapter is trained on, as PEFT records it.
	BaseModel string
	// RSLoRA scales every module by Alpha over the root of its rank rather than
	// over its rank (rank-stabilized LoRA), as PEFT records with use_rslora.
	RSLoRA bool
	// UpRate is LoRA+: the up factor learns this many times faster than the down
	// factor; zero or one is plain LoRA. The optimizer steps every factor by a
	// size that does not depend on its scale, so the up factor is stored over the
	// rate and its module's alpha is the rate times Alpha: the product, and so the
	// adapted model, is the one a faster up factor would reach, and a saved
	// adapter's alpha is that of the module as trained.
	UpRate float64
}

// Trainer trains one adapter over a model. It owns the model's adapted weights
// while it lives: they hold the base plus the adapter's update until RestoreBase.
type Trainer struct {
	model     *densecausal.Model
	adapter   peft.Adapter
	frozen    map[string][]float32
	pack      *optimizer.TensorPack
	optimizer *optimizer.Optimizer
	config    optimizer.Config
	gradients map[string][]float32
	// quantized is set when the model holds the adapted weights frozen in their stored
	// quantization, which take the update as each layer reads them (Model.OnDequantize).
	quantized bool
	// average, once Average starts it, follows the factors after each update.
	average *optimizer.Average
	// layerModules are the adapter's modules by the decoder layer they adapt.
	layerModules map[int][]string
	// onSettle, when set, is told how many full-size gradient values a step holds as each
	// layer's are projected: a test's view of the step's peak.
	onSettle func(live int)
}

// New starts an adapter on the model: every target projection of every dense
// layer gets a pair whose down factor is drawn uniformly from ±1/√inputs, as
// PEFT initializes it, and whose up factor is zero, so the adapted model begins
// equal to the base.
func New(model *densecausal.Model, config Config) (*Trainer, error) {
	if model == nil || config.Rank <= 0 || math.IsNaN(config.Alpha) || math.IsInf(config.Alpha, 0) || config.Alpha <= 0 {
		return nil, errors.New("loratrain: a model and a positive rank and alpha are required")
	}
	if config.UpRate == 0 {
		config.UpRate = 1
	}
	if math.IsNaN(config.UpRate) || math.IsInf(config.UpRate, 0) || config.UpRate < 1 {
		return nil, fmt.Errorf("loratrain: an up-factor rate of %v is below one, where the up factor would learn slower than the down factor", config.UpRate)
	}
	// The scale each module trains under: LoRA+ folds its rate into alpha (see Config.UpRate).
	config.Alpha *= config.UpRate
	targets := config.Targets
	if len(targets) == 0 {
		for _, projected := range projections {
			targets = append(targets, projected.target)
		}
	}
	random := rand.New(rand.NewPCG(config.Seed, config.Seed))
	modules := map[string]peft.Module{}
	targetOf := map[string]string{}
	for layer := range model.Dims.Layers {
		names := densecausal.DenseLayerTensorNames(layer)
		for _, projected := range projections {
			targetOf[projected.weight(names)] = projected.target
		}
	}
	for _, name := range slices.Sorted(maps.Keys(targetOf)) {
		shape := model.Shapes[name]
		if !slices.Contains(targets, targetOf[name]) || len(shape) != matrixRank {
			continue
		}
		module := strings.TrimSuffix(name, weightSuffix)
		out, in := shape[0], shape[1]
		if config.Rank > min(in, out) {
			return nil, fmt.Errorf("loratrain: rank %d exceeds the %d by %d projection %s", config.Rank, out, in, module)
		}
		bound := 1 / math.Sqrt(float64(in))
		down := make([]float32, config.Rank*in)
		for index := range down {
			down[index] = float32((uniformSpan*random.Float64() - 1) * bound)
		}
		modules[module] = peft.Module{Rank: config.Rank, In: in, Out: out, Alpha: config.Alpha, Down: down, Up: make([]float32, out*config.Rank)}
	}
	if len(modules) == 0 {
		return nil, fmt.Errorf("loratrain: no dense layer of the model has a target projection among %v", targets)
	}
	adapter := peft.Adapter{
		Config:  peft.Config{BaseModel: config.BaseModel, Rank: config.Rank, Alpha: config.Alpha, RSLoRA: config.RSLoRA, TargetModules: slices.Compact(slices.Sorted(slices.Values(targets)))},
		Modules: modules,
	}
	return bind(model, adapter)
}

// Resume continues an existing adapter, such as one loaded from a PEFT
// directory: its factors are copied, the model's adapted weights take its
// update, and the optimizer's momentum starts empty.
func Resume(model *densecausal.Model, adapter peft.Adapter) (*Trainer, error) {
	if model == nil {
		return nil, errors.New("loratrain: a model is required")
	}
	if err := adapter.Validate(); err != nil {
		return nil, err
	}
	copied := peft.Adapter{Config: adapter.Config, Modules: make(map[string]peft.Module, len(adapter.Modules))}
	for name, module := range adapter.Modules {
		module.Down, module.Up = slices.Clone(module.Down), slices.Clone(module.Up)
		copied.Modules[name] = module
	}
	return bind(model, copied)
}

// bind freezes the model's adapted weights and packs the adapter's factors for
// the optimizer.
func bind(model *densecausal.Model, adapter peft.Adapter) (*Trainer, error) {
	trainer := &Trainer{model: model, adapter: adapter, frozen: map[string][]float32{}, gradients: map[string][]float32{}, layerModules: map[int][]string{}}
	factors := map[string][]float32{}
	shapes := map[string][matrixRank]int{}
	for name, module := range adapter.Modules {
		if layer, layered := moduleLayer(name); layered {
			trainer.layerModules[layer] = append(trainer.layerModules[layer], name)
		}
		weight := name + weightSuffix
		shape := model.Shapes[weight]
		if len(shape) != matrixRank || shape[0] != module.Out || shape[1] != module.In {
			return nil, fmt.Errorf("loratrain: adapter module %q is %d by %d, the model's %s is %v", name, module.Out, module.In, weight, shape)
		}
		if model.Frozen(weight) {
			// The model dequantizes this weight for each layer that reads it and hands it here to
			// take the update, so no copy of the base or of the adapted weight is held.
			trainer.quantized = true
		} else {
			if len(model.Weights[weight]) != module.Out*module.In {
				return nil, fmt.Errorf("loratrain: the model's %s holds %d values, adapter module %q is %d by %d", weight, len(model.Weights[weight]), name, module.Out, module.In)
			}
			trainer.frozen[name] = slices.Clone(model.Weights[weight])
		}
		factors[name+downFactor], shapes[name+downFactor] = module.Down, [matrixRank]int{module.Rank, module.In}
		factors[name+upFactor], shapes[name+upFactor] = module.Up, [matrixRank]int{module.Out, module.Rank}
	}
	pack, err := optimizer.NewTensorPack(factors, optimizer.MatrixGeometry(shapes))
	if err != nil {
		return nil, err
	}
	trainer.pack = pack
	if trainer.quantized {
		if err := model.OnDequantize(trainer.addUpdate); err != nil {
			return nil, err
		}
	}
	return trainer, trainer.apply()
}

// addUpdate adds the adapter's update to a frozen weight the model has just
// dequantized for a layer; a weight no module adapts is left as it is.
func (trainer *Trainer) addUpdate(weight string, values []float32) error {
	module, isWeight := strings.CutSuffix(weight, weightSuffix)
	if _, adapted := trainer.adapter.Modules[module]; !isWeight || !adapted {
		return nil
	}
	return trainer.adapter.AddDelta(values, module)
}

// apply puts the base plus the adapter's update into each adapted weight, in
// place: the model reads its weights through slices it bound when it loaded.
func (trainer *Trainer) apply() error {
	for name := range trainer.adapter.Modules {
		base, held := trainer.frozen[name]
		if !held {
			continue // a frozen weight takes the update as each layer reads it
		}
		target := trainer.model.Weights[name+weightSuffix]
		copy(target, base)
		if err := trainer.adapter.AddDelta(target, name); err != nil {
			return err
		}
	}
	return nil
}

// Unmerged is the base values of the adapted weights a model that holds others of
// them quantized holds in full precision and has adapted in place (a projection the
// file stores in full precision, or a llama query or key, which loads reordered): the
// model a quantized run publishes is the base as stored, the adapter beside it, so
// these are what it publishes of them. It is nil for a run that holds none quantized.
func (trainer *Trainer) Unmerged() map[string][]float32 {
	if !trainer.quantized {
		return nil
	}
	unmerged := make(map[string][]float32, len(trainer.frozen))
	for name, base := range trainer.frozen {
		unmerged[name+weightSuffix] = base
	}
	return unmerged
}

// RestoreBase puts the model's adapted weights back to the frozen base: the
// copies of them it holds, and for weights frozen in their quantization, the
// update the model adds as each layer reads them, which it stops adding.
func (trainer *Trainer) RestoreBase() {
	for name, base := range trainer.frozen {
		copy(trainer.model.Weights[name+weightSuffix], base)
	}
	if trainer.quantized {
		_ = trainer.model.OnDequantize(nil)
	}
}

// Adapts is the predicate for the weights an adapter over the projections named
// by targets (the seven linear ones when empty) adapts, which a model can hold
// frozen in their stored quantization (densecausal.LoadGGUFFrozen).
func Adapts(targets []string) func(weight string) bool {
	if len(targets) == 0 {
		for _, projected := range projections {
			targets = append(targets, projected.target)
		}
	}
	return func(weight string) bool {
		rest, prefixed := strings.CutPrefix(weight, layerNamePrefix)
		layer, _, nested := strings.Cut(rest, ".")
		index, err := strconv.Atoi(layer)
		if !prefixed || !nested || err != nil {
			return false
		}
		names := densecausal.DenseLayerTensorNames(index)
		for _, projected := range projections {
			if projected.weight(names) == weight && slices.Contains(targets, projected.target) {
				return true
			}
		}
		return false
	}
}

// layerNamePrefix opens the name of every weight of a decoder layer, followed by
// the layer's number.
const layerNamePrefix = "model.layers."

// Adapter returns the adapter as it stands now. Its factors are the trainer's
// own, not copies: save or clone it before training further.
func (trainer *Trainer) Adapter() peft.Adapter { return trainer.adapter }

// Plan is the optimizer plan over the adapter's factors, the parameters a
// training step changes.
func (trainer *Trainer) Plan() optimizer.Plan { return trainer.pack.Plan() }

// OptimizerState is the optimizer's progress and momentum, which pair with Plan.
func (trainer *Trainer) OptimizerState() (optimizer.State, error) {
	if trainer.optimizer == nil {
		return optimizer.State{}, errors.New("loratrain: no step has been taken")
	}
	return trainer.optimizer.Snapshot(), nil
}

// Step takes one update on one token batch and returns its loss: the adapted
// model's, before the update, over the tokens trained marks (every token when
// it is nil), as densecausal's MaskedLossAndGrads learns them.
func (trainer *Trainer) Step(tokens []int, trained []bool, learningRate, momentum float64) (float64, error) {
	if err := trainer.ensureOptimizer(learningRate, momentum); err != nil {
		return 0, err
	}
	// Each layer's weight gradients are projected to its modules' factor gradients as that layer's
	// backward finishes and released, so a step holds one layer's full-size gradients, not every
	// layer's; a module outside the layers is projected from what the pass returns.
	settled := map[string]bool{}
	loss, grads, err := trainer.model.MaskedLossAndLayerGradsOf(tokens, trained, trainer.adapted, func(layer int, gradients densecausal.Grads) error {
		if trainer.onSettle != nil {
			trainer.onSettle(liveValues(gradients))
		}
		for _, name := range trainer.layerModules[layer] {
			if err := trainer.project(name, gradients); err != nil {
				return err
			}
			settled[name] = true
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	for name := range trainer.adapter.Modules {
		if settled[name] {
			continue
		}
		if err := trainer.project(name, grads); err != nil {
			return 0, err
		}
	}
	if err := trainer.pack.GatherGradients(trainer.gradients); err != nil {
		return 0, err
	}
	trainer.optimizer.Step()
	trainer.pack.Scatter()
	if trainer.average != nil {
		factors, _ := trainer.pack.Slabs()
		if err := trainer.average.Update(factors); err != nil {
			return 0, err
		}
	}
	return loss, trainer.apply()
}

// Average starts a moving average of the factors under the decay (see
// optimizer.Average), taken after each update from here on, from the factors as
// they stand. UseAverage then makes the adapter the average.
func (trainer *Trainer) Average(decay float64) error {
	factors, _ := trainer.pack.Slabs()
	average, err := optimizer.NewAverage(factors, decay)
	if err != nil {
		return err
	}
	trainer.average = average
	return nil
}

// AverageState is the moving average of the factors as it stands, nil before
// Average starts one; it is the trainer's own, which a checkpoint copies.
func (trainer *Trainer) AverageState() *optimizer.Average { return trainer.average }

// Restore continues training from the optimizer state and moving average (nil
// for none) an earlier run left at its update, under the rate and momentum it
// took them with: the trainer must be over the adapter that run left, which
// Resume made, so that the next update is the one the earlier run took next.
func (trainer *Trainer) Restore(state optimizer.State, average *optimizer.Average, learningRate, momentum float64) error {
	if err := trainer.ensureOptimizer(learningRate, momentum); err != nil {
		return err
	}
	if err := trainer.optimizer.Restore(state); err != nil {
		return err
	}
	factors, _ := trainer.pack.Slabs()
	if average != nil && len(average.Values()) != len(factors) {
		return fmt.Errorf("loratrain: a resumed average spans %d factor values, the adapter's are %d", len(average.Values()), len(factors))
	}
	trainer.average = average
	return nil
}

// UseAverage makes the adapter, and the model it is applied to, the moving
// average of the factors Average started, leaving the optimizer's state as the
// updates took it.
func (trainer *Trainer) UseAverage() error {
	if trainer.average == nil {
		return errors.New("loratrain: no average was started")
	}
	factors, _ := trainer.pack.Slabs()
	copy(factors, trainer.average.Values())
	trainer.pack.Scatter()
	return trainer.apply()
}

// Train takes one update per batch and returns each update's loss. trained
// holds each batch's flags for the tokens it learns, or is nil to learn every
// token. The observer sees every finished update; an error from it stops
// training there.
func (trainer *Trainer) Train(batches [][]int, trained [][]bool, learningRate, momentum float64, observe densecausal.TrainObserver) ([]float64, error) {
	if len(batches) == 0 {
		return nil, errors.New("loratrain: token batches absent")
	}
	if trained != nil && len(trained) != len(batches) {
		return nil, fmt.Errorf("loratrain: trained flags for %d batches, %d given", len(batches), len(trained))
	}
	losses := make([]float64, 0, len(batches))
	for step, tokens := range batches {
		stopwatch := processmeasure.NewStopwatch()
		var flags []bool
		if trained != nil {
			flags = trained[step]
		}
		loss, err := trainer.Step(tokens, flags, learningRate, momentum)
		if err != nil {
			return losses, err
		}
		losses = append(losses, loss)
		if observe != nil {
			wall, err := stopwatch.Elapsed()
			if err != nil {
				return losses, err
			}
			if err := observe(step, loss, time.Duration(wall)); err != nil {
				return losses, err
			}
		}
	}
	return losses, nil
}

// adapted reports whether the weight name is one the adapter adapts, the only
// weights whose gradient a step needs.
func (trainer *Trainer) adapted(weight string) bool {
	module, isWeight := strings.CutSuffix(weight, weightSuffix)
	_, adapted := trainer.adapter.Modules[module]
	return isWeight && adapted
}

// ensureOptimizer creates the optimizer on the first update. Its rate and
// momentum are the run's, so a later update under others is refused rather than
// quietly stepped under the first.
func (trainer *Trainer) ensureOptimizer(learningRate, momentum float64) error {
	config := optimizer.Config{BaseLearningRate: learningRate, Momentum: momentum, Schedule: optimizer.ScheduleConstant}
	if trainer.optimizer == nil {
		created, err := trainer.pack.NewOptimizer(config)
		if err != nil {
			return err
		}
		trainer.optimizer, trainer.config = created, config
		return nil
	}
	if config != trainer.config {
		return errors.New("loratrain: the learning rate and momentum are fixed by the first update")
	}
	return nil
}

// project turns the gradient of the loss with respect to the effective weight of
// one module into the gradients of its two factors: with W = base + s·B·A,
// dB = s·dW·Aᵀ and dA = s·Bᵀ·dW.
func (trainer *Trainer) project(name string, grads densecausal.Grads) error {
	module := trainer.adapter.Modules[name]
	gradient := grads[name+weightSuffix]
	if len(gradient) != module.Out*module.In {
		return fmt.Errorf("loratrain: the backward pass gave %d gradient values for %s, want %d", len(gradient), name+weightSuffix, module.Out*module.In)
	}
	scale, err := trainer.adapter.Scale(name)
	if err != nil {
		return err
	}
	up := make([]float32, module.Out*module.Rank)
	hostmath.Linear(up, gradient, module.Down, module.Out, module.In, module.Rank)
	down := make([]float32, module.Rank*module.In)
	hostmath.LinearBackwardInput(down, hostmath.Transpose2D(module.Up, module.Out, module.Rank), gradient, module.Rank, module.In, module.Out)
	for _, factor := range [][]float32{up, down} {
		for index := range factor {
			factor[index] *= float32(scale)
		}
	}
	trainer.gradients[name+downFactor], trainer.gradients[name+upFactor] = down, up
	return nil
}

// liveValues is how many gradient values the map holds.
func liveValues(gradients densecausal.Grads) int {
	total := 0
	for _, gradient := range gradients {
		total += len(gradient)
	}
	return total
}

// moduleLayer is the decoder layer a module of the adapter belongs to, which its
// name gives as model.layers.<n>.<module>.
func moduleLayer(module string) (int, bool) {
	rest, prefixed := strings.CutPrefix(module, layerNamePrefix)
	number, _, nested := strings.Cut(rest, ".")
	layer, err := strconv.Atoi(number)
	return layer, prefixed && nested && err == nil
}
