package inference

import (
	"context"
	"errors"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
)

// projectFeatures: execute the compiled feature projection.
func (r *Runner) projectFeatures(ctx context.Context, features reference.Value) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errRunnerNil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.weights.FeatureProjection == nil {
		return reference.Value{}, errors.New("inference: feature projection is unavailable")
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	operands := model.ProjectionOperands{Input: runtime.input("features.input", features)}
	var err error
	operands.Primary, err = runtime.weight(*r.weights.FeatureProjection)
	if err != nil {
		return reference.Value{}, err
	}
	if r.weights.EncoderOutputNorm != nil {
		operands.Normalization, err = runtime.weight(*r.weights.EncoderOutputNorm)
		if err != nil {
			return reference.Value{}, err
		}
	}
	result, err := r.program.Model.Projection(model.ProjectionFeature).Build(runtime.builder, operands)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(result.Primary)
	if err != nil {
		return reference.Value{}, err
	}
	return results[result.Primary], nil
}
