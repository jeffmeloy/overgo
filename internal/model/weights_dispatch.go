package model

import "overgo/internal/gguf"

// ReadWeights: profile-routed tensor catalog validation.
func ReadWeights(file *gguf.File, spec Spec) (Weights, error) {
	_, ok := spec.ResolvedProfile()
	if !ok {
		return Weights{}, &UnsupportedArchitectureError{
			Architecture: spec.Architecture,
		}
	}
	return readWeightCatalog(file, spec)
}
