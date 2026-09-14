package trainingprogram

import (
	"context"
	_ "embed"
	"errors"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
	"overgo/internal/recipe"
)

const (
	optimizerPolicyMedia  = "application/vnd.overgo.optimizer-policy+json"
	optimizerPolicySchema = "overgo/optimizer-policy/v1"
)

type OptimizerSchedule string

const OptimizerScheduleConstant OptimizerSchedule = "constant"

type OptimizerPolicy struct {
	ID                       artifact.ID       `json:"-"`
	ParameterExponent        float64           `json:"parameter_exponent"`
	MomentumEffectiveSamples int               `json:"momentum_effective_samples"`
	Schedule                 OptimizerSchedule `json:"schedule"`
}

var optimizerPolicyCodec = artifact.JSONDocumentCodec(
	"optimizer policy", artifact.KindProfile, optimizerPolicyMedia, optimizerPolicySchema,
	validateOptimizerPolicy, func(value OptimizerPolicy) artifact.ID { return value.ID },
	func(value *OptimizerPolicy, id artifact.ID) { value.ID = id }, nil,
)

//go:embed optimizer_policy.json
var builtinOptimizerPolicyJSON []byte

var builtinOptimizerPolicy = func() OptimizerPolicy {
	policy, _, err := optimizerPolicyCodec.Normalize(builtinOptimizerPolicyJSON)
	if err != nil {
		panic(err)
	}
	return policy
}()

func BuiltinOptimizerPolicy() OptimizerPolicy { return builtinOptimizerPolicy }

func (policy OptimizerPolicy) Content() (artifact.Content, error) {
	return optimizerPolicyCodec.Content(policy)
}

func RequireOptimizerPolicy(ctx context.Context, reader artifact.Reader, id artifact.ID) (OptimizerPolicy, error) {
	return optimizerPolicyCodec.Require(ctx, reader, id)
}

func OptimizerPolicyFromRecipe(ctx context.Context, reader artifact.Reader, definition recipe.Definition) (OptimizerPolicy, error) {
	id, ok := definition.PrimaryDependency(recipe.DependencyOptimizer)
	if !ok {
		return OptimizerPolicy{}, errors.New("training program: optimizer policy dependency absent")
	}
	return RequireOptimizerPolicy(ctx, reader, id)
}

func (policy OptimizerPolicy) BaseLearningRate(parameters int) float64 {
	if parameters <= 0 {
		return 0
	}
	return math.Pow(float64(parameters), policy.ParameterExponent)
}

func (policy OptimizerPolicy) Momentum() float64 {
	samples := float64(policy.MomentumEffectiveSamples)
	return (samples - 1) / (samples + 1)
}

func (policy OptimizerPolicy) Config(parameters int) (hostoptimizer.Config, error) {
	if err := optimizerPolicyCodec.ValidateIdentity(policy); err != nil {
		return hostoptimizer.Config{}, err
	}
	if parameters <= 0 {
		return hostoptimizer.Config{}, errors.New("training program: optimizer policy requires positive parameters")
	}
	return hostoptimizer.Config{
		BaseLearningRate: policy.BaseLearningRate(parameters), Momentum: policy.Momentum(),
		Schedule: hostoptimizer.ScheduleConstant,
	}, nil
}

func validateOptimizerPolicy(policy *OptimizerPolicy) error {
	if policy == nil || !finite(policy.ParameterExponent) || policy.ParameterExponent >= 0 ||
		policy.MomentumEffectiveSamples <= 1 || policy.Schedule != OptimizerScheduleConstant {
		return errors.New("training program: invalid optimizer policy")
	}
	return nil
}
