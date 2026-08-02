package model

import "llamacpp2go/internal/gguf"

// ReadWeights: family-routed tensor catalog validation.
func ReadWeights(file *gguf.File, spec Spec) (Weights, error) {
	if spec.Architecture == "" {
		return readWeightCatalog(file, spec)
	}
	_, ok := LookupArchitecture(spec.Architecture)
	if !ok {
		return Weights{}, &UnsupportedArchitectureError{
			Architecture: spec.Architecture,
		}
	}
	return readWeightCatalog(file, spec)
}
