package model

import (
	"errors"

	"overgo/internal/tensor"
)

// BuildFeatureProjection executes the compiled feature-fusion policy.
func (p ModelPlan) BuildFeatureProjection(
	builder *tensor.Builder,
	features, projection, norm *tensor.Tensor,
) (*tensor.Tensor, error) {
	var width uint64
	normalize := false
	switch p.profile.Forward {
	case ForwardEagle3:
		width = uint64(eagle3TargetLayerCount) * uint64(p.spec.TargetHiddenSize)
	case ForwardDFlash:
		width = uint64(len(p.spec.TargetLayers)) * uint64(p.spec.EmbeddingLength)
		normalize = true
	}
	if builder == nil || features == nil || projection == nil ||
		width == 0 || features.Shape.Rank != 2 || features.Shape.Dims[0] != width ||
		normalize != (norm != nil) {
		return nil, errors.New("compiled feature projection is incompatible")
	}
	output := builder.MulMat(projection, features)
	if normalize {
		output = builder.WeightedRMSNorm(output, norm, p.spec.RMSNormEpsilon)
	}
	return output, builder.Err()
}
