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
	Plan             *LayerPlan
}

// BuildArchitectureBlockCached: family-routed graph construction.
func BuildArchitectureBlockCached(
	options BlockDispatchOptions,
) (DenseBlockResult, error) {
	_, ok := LookupArchitecture(options.Spec.Architecture)
	if !ok {
		return DenseBlockResult{}, &UnsupportedArchitectureError{
			Architecture: options.Spec.Architecture,
		}
	}
	plan := options.Spec.PlanLayer(options.Layer, options.Recurrent)
	if options.Plan != nil {
		plan = *options.Plan
	}
	if plan.GraphFamily == ArchitectureFamilyEncoderDecoder {
		return DenseBlockResult{}, errors.New(
			"encoder-decoder blocks require explicit encoder state",
		)
	}
	switch plan.Block {
	case BlockMamba:
		return BuildMambaBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case BlockMamba2:
		return BuildMamba2BlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case BlockFalconH1:
		return BuildFalconH1BlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue,
			options.PastConvState, options.PastSSMState,
		)
	case BlockJamba:
		return BuildJambaRecurrentBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case BlockGraniteHybrid:
		return BuildGraniteHybridRecurrentBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case BlockPLaMo2:
		return BuildPLaMo2RecurrentBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.PastKey, options.PastValue,
		)
	case BlockNemotronH:
		return BuildNemotronHBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue, options.Layer,
		)
	case BlockKimiLinear:
		return BuildKimiLinearBlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, plan.Recurrent, options.PastKey,
			options.PastValue, options.Layer,
		)
	case BlockDSA:
		return BuildDSABlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue,
			options.PastIndexerKey, options.PerLayerInput, options.Layer,
		)
	case BlockDeepSeek4:
		return BuildDeepSeek4BlockCached(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.TokenRows, options.PastKey,
			options.PastStates, options.CurrentPositions, options.Layer,
		)
	case BlockMLA:
		return BuildMLABlockCachedForLayer(
			options.Builder, options.Input, options.Spec, options.Weights,
			options.Positions, options.PastKey, options.PastValue, options.Layer,
		)
	case BlockDense:
		return buildAttentionFamilyBlock(options)
	default:
		return DenseBlockResult{}, errors.New("unknown compiled block policy")
	}
}

func buildAttentionFamilyBlock(
	options BlockDispatchOptions,
) (DenseBlockResult, error) {
	return BuildDenseBlockWithOptions(DenseBlockOptions{
		Builder: options.Builder, Input: options.Input, Spec: options.Spec,
		Weights: options.Weights, Positions: options.Positions,
		MultiPositions: options.MultiPositions, PastKey: options.PastKey,
		PastValue: options.PastValue, Layer: options.Layer, Plan: options.Plan,
	})
}
