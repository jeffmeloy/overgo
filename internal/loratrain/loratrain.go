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

	"overgo/internal/binaryschema"
	"overgo/internal/densecausal"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/peft"
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
	// layerPrefix, projectionSuffix and the two block names make up a dense
	// decoder layer's linear projection weight: model.layers.N.<block>.<name>_proj.weight.
	layerPrefix      = "model.layers."
	projectionSuffix = "_proj"
	attentionBlock   = "self_attn"
	mlpBlock         = "mlp"
)

// projection splits a dense decoder layer's linear projection weight name into
// the module path PEFT names it by and the projection's own name, as
// "model.layers.0.self_attn.q_proj.weight" into "model.layers.0.self_attn.q_proj"
// and "q_proj". Anything else, such as a mixture expert's weight, which sits one
// path segment deeper, is not one.
func projection(weight string) (module, name string, ok bool) {
	rest, prefixed := strings.CutPrefix(weight, layerPrefix)
	module, found := strings.CutSuffix(weight, weightSuffix)
	if !prefixed || !found {
		return "", "", false
	}
	layer, rest, prefixed := strings.Cut(strings.TrimSuffix(rest, weightSuffix), ".")
	if _, err := strconv.ParseUint(layer, binaryschema.DecimalRadix, binaryschema.Width32Bits); err != nil || !prefixed {
		return "", "", false
	}
	block, name, nested := strings.Cut(rest, ".")
	if !nested || (block != attentionBlock && block != mlpBlock) || !strings.HasSuffix(name, projectionSuffix) || strings.Contains(name, ".") {
		return "", "", false
	}
	return module, name, true
}

// defaultTargets are the seven linear projections of a dense decoder layer,
// the layers a LoRA adapter on every linear layer adapts.
var defaultTargets = []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

// Config names the adapter a fresh training starts.
type Config struct {
	// Rank and Alpha are every module's: its scale is Alpha over Rank.
	Rank  int
	Alpha float64
	// Targets are the projection names to adapt; empty adapts defaultTargets.
	Targets []string
	// Seed makes the factors' initialization repeat.
	Seed uint64
	// BaseModel names the model the adapter is trained on, as PEFT records it.
	BaseModel string
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
	// average, once Average starts it, follows the factors after each update.
	average *optimizer.Average
}

// New starts an adapter on the model: every target projection of every dense
// layer gets a pair whose down factor is drawn uniformly from ±1/√inputs, as
// PEFT initializes it, and whose up factor is zero, so the adapted model begins
// equal to the base.
func New(model *densecausal.Model, config Config) (*Trainer, error) {
	if model == nil || config.Rank <= 0 || math.IsNaN(config.Alpha) || math.IsInf(config.Alpha, 0) || config.Alpha <= 0 {
		return nil, errors.New("loratrain: a model and a positive rank and alpha are required")
	}
	targets := config.Targets
	if len(targets) == 0 {
		targets = defaultTargets
	}
	random := rand.New(rand.NewPCG(config.Seed, config.Seed))
	modules := map[string]peft.Module{}
	for _, name := range slices.Sorted(maps.Keys(model.Weights)) {
		module, projected, isProjection := projection(name)
		shape := model.Shapes[name]
		if !isProjection || !slices.Contains(targets, projected) || len(shape) != matrixRank {
			continue
		}
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
		Config:  peft.Config{BaseModel: config.BaseModel, Rank: config.Rank, Alpha: config.Alpha, TargetModules: slices.Compact(slices.Sorted(slices.Values(targets)))},
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
	trainer := &Trainer{model: model, adapter: adapter, frozen: map[string][]float32{}, gradients: map[string][]float32{}}
	factors := map[string][]float32{}
	shapes := map[string][matrixRank]int{}
	for name, module := range adapter.Modules {
		weight := name + weightSuffix
		shape := model.Shapes[weight]
		if len(shape) != matrixRank || shape[0] != module.Out || shape[1] != module.In || len(model.Weights[weight]) != module.Out*module.In {
			return nil, fmt.Errorf("loratrain: adapter module %q is %d by %d, the model's %s is %v", name, module.Out, module.In, weight, shape)
		}
		trainer.frozen[name] = slices.Clone(model.Weights[weight])
		factors[name+downFactor], shapes[name+downFactor] = module.Down, [matrixRank]int{module.Rank, module.In}
		factors[name+upFactor], shapes[name+upFactor] = module.Up, [matrixRank]int{module.Out, module.Rank}
	}
	pack, err := optimizer.NewTensorPack(factors, optimizer.MatrixGeometry(shapes))
	if err != nil {
		return nil, err
	}
	trainer.pack = pack
	return trainer, trainer.apply()
}

// apply puts the base plus the adapter's update into each adapted weight, in
// place: the model reads its weights through slices it bound when it loaded.
func (trainer *Trainer) apply() error {
	for name := range trainer.adapter.Modules {
		delta, err := trainer.adapter.Delta(name)
		if err != nil {
			return err
		}
		target, base := trainer.model.Weights[name+weightSuffix], trainer.frozen[name]
		for index := range target {
			target[index] = base[index] + delta[index]
		}
	}
	return nil
}

// RestoreBase puts the model's adapted weights back to the frozen base.
func (trainer *Trainer) RestoreBase() {
	for name, base := range trainer.frozen {
		copy(trainer.model.Weights[name+weightSuffix], base)
	}
}

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
	loss, grads, err := trainer.model.MaskedLossAndGrads(tokens, trained)
	if err != nil {
		return 0, err
	}
	if err := trainer.project(grads); err != nil {
		return 0, err
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
		started := time.Now()
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
			if err := observe(step, loss, time.Since(started)); err != nil {
				return losses, err
			}
		}
	}
	return losses, nil
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

// project turns the gradient of the loss with respect to each effective weight
// into the gradients of its two factors: with W = base + s·B·A, dB = s·dW·Aᵀ
// and dA = s·Bᵀ·dW.
func (trainer *Trainer) project(grads densecausal.Grads) error {
	for name, module := range trainer.adapter.Modules {
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
		hostmath.Linear(down, hostmath.Transpose2D(module.Up, module.Out, module.Rank), hostmath.Transpose2D(gradient, module.Out, module.In), module.Rank, module.Out, module.In)
		for _, factor := range [][]float32{up, down} {
			for index := range factor {
				factor[index] *= float32(scale)
			}
		}
		trainer.gradients[name+downFactor], trainer.gradients[name+upFactor] = down, up
	}
	return nil
}
