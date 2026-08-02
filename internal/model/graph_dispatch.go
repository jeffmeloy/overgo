package model

import (
	"errors"

	"llamacpp2go/internal/tensor"
)

// BlockDispatchOptions: family-dispatch graph inputs.
type BlockDispatchOptions struct {
	Builder          *tensor.Builder
	Input            *tensor.Tensor
	Spec             Spec
	Weights          LayerGraphWeights
	Positions        []uint32
	MultiPositions   *[4][]uint32
	TokenRows        []uint32
	PastKey          *tensor.Tensor
	PastValue        *tensor.Tensor
	PastIndexerKey   *tensor.Tensor
	PastConvState    *tensor.Tensor
	PastSSMState     *tensor.Tensor
	PastStates       map[string]*tensor.Tensor
	CurrentPositions *tensor.Tensor
	PerLayerInput    *tensor.Tensor
	Layer            uint32
	Recurrent        bool
}

// BuildArchitectureBlockCached: family-routed graph construction.
func BuildArchitectureBlockCached(
	options BlockDispatchOptions,
) (DenseBlockResult, error) {
	profile, ok := LookupArchitecture(options.Spec.Architecture)
	if !ok {
		return DenseBlockResult{}, &UnsupportedArchitectureError{
			Architecture: options.Spec.Architecture,
		}
	}
	switch profile.GraphFamily {
	case ArchitectureFamilyRecurrent, ArchitectureFamilyHybrid:
		if result, handled, err := buildRecurrentFamilyBlock(options); handled {
			return result, err
		}
	case ArchitectureFamilyMoE:
		if result, handled, err := buildMoEFamilyBlock(options); handled {
			return result, err
		}
	case ArchitectureFamilyEncoderDecoder:
		return DenseBlockResult{}, errors.New(
			"encoder-decoder blocks require explicit encoder state",
		)
	}
	if profile.Has(ArchitectureMLA) {
		return BuildMLABlockCachedForLayer(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue, options.Layer,
		)
	}
	return buildAttentionFamilyBlock(options)
}

func buildRecurrentFamilyBlock(
	options BlockDispatchOptions,
) (DenseBlockResult, bool, error) {
	var result DenseBlockResult
	var err error
	switch options.Spec.Architecture {
	case "mamba":
		result, err = BuildMambaBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case "mamba2":
		result, err = BuildMamba2BlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case "falcon-h1":
		result, err = BuildFalconH1BlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue,
			options.PastConvState, options.PastSSMState,
		)
	case "jamba":
		if !options.Recurrent {
			return DenseBlockResult{}, false, nil
		}
		result, err = BuildJambaRecurrentBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case "granitehybrid":
		if !options.Recurrent {
			return DenseBlockResult{}, false, nil
		}
		result, err = BuildGraniteHybridRecurrentBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case "plamo2":
		if !options.Recurrent {
			return DenseBlockResult{}, false, nil
		}
		result, err = BuildPLaMo2RecurrentBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case "nemotron_h", "nemotron_h_moe":
		result, err = BuildNemotronHBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue, options.Layer,
		)
	case "kimi-linear":
		result, err = BuildKimiLinearBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.Recurrent, options.PastKey,
			options.PastValue, options.Layer,
		)
	default:
		return DenseBlockResult{}, false, nil
	}
	return result, true, err
}

func buildMoEFamilyBlock(
	options BlockDispatchOptions,
) (DenseBlockResult, bool, error) {
	profile := options.Spec.Profile()
	if profile.Has(ArchitectureDSA) {
		result, err := BuildDSABlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue,
			options.PastIndexerKey, options.PerLayerInput, options.Layer,
		)
		return result, true, err
	}
	if options.Spec.Architecture == "deepseek4" {
		result, err := BuildDeepSeek4BlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.TokenRows, options.PastKey,
			options.PastStates, options.CurrentPositions, options.Layer,
		)
		return result, true, err
	}
	if profile.Has(ArchitectureMLA) {
		result, err := BuildMLABlockCachedForLayer(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue, options.Layer,
		)
		return result, true, err
	}
	return DenseBlockResult{}, false, nil
}

func buildAttentionFamilyBlock(
	options BlockDispatchOptions,
) (DenseBlockResult, error) {
	if options.MultiPositions != nil {
		return BuildDenseBlockCachedForLayerWithMultiPositions(
			options.Builder, options.Input, options.Spec, options.Weights,
			*options.MultiPositions, options.PastKey, options.PastValue,
			options.Layer,
		)
	}
	return BuildDenseBlockCachedForLayer(
		options.Builder, options.Input, options.Spec, options.Weights,
		options.Positions, options.PastKey, options.PastValue, options.Layer,
	)
}
