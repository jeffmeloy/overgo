package model

import "llamacpp2go/internal/gguf"

// ReadWeights: family-routed tensor catalog validation.
func ReadWeights(file *gguf.File, spec Spec) (Weights, error) {
	if spec.Architecture == "" {
		return readWeightCatalog(file, spec)
	}
	profile, ok := LookupArchitecture(spec.Architecture)
	if !ok {
		return Weights{}, &UnsupportedArchitectureError{
			Architecture: spec.Architecture,
		}
	}
	switch profile.CatalogFamily {
	case ArchitectureFamilyAttention:
		return readAttentionWeightCatalog(file, spec)
	case ArchitectureFamilyMoE:
		return readMoEWeightCatalog(file, spec)
	case ArchitectureFamilyRecurrent:
		return readRecurrentWeightCatalog(file, spec)
	case ArchitectureFamilyHybrid:
		return readHybridWeightCatalog(file, spec)
	case ArchitectureFamilyEncoder:
		return readEncoderWeightCatalog(file, spec)
	case ArchitectureFamilyEncoderDecoder:
		return readEncoderDecoderWeightCatalog(file, spec)
	case ArchitectureFamilyDiffusion:
		return readDiffusionWeightCatalog(file, spec)
	case ArchitectureFamilyDraft:
		return readDraftWeightCatalog(file, spec)
	default:
		return Weights{}, &UnsupportedArchitectureError{
			Architecture: spec.Architecture,
		}
	}
}

func readAttentionWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	return readWeightCatalog(file, spec)
}

func readMoEWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	return readWeightCatalog(file, spec)
}

func readRecurrentWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	return readWeightCatalog(file, spec)
}

func readHybridWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	return readWeightCatalog(file, spec)
}

func readEncoderWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	return readWeightCatalog(file, spec)
}

func readEncoderDecoderWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	return readWeightCatalog(file, spec)
}

func readDiffusionWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	return readWeightCatalog(file, spec)
}

func readDraftWeightCatalog(file *gguf.File, spec Spec) (Weights, error) {
	return readWeightCatalog(file, spec)
}
