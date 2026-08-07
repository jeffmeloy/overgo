package model

import (
	"errors"

	"overgo/internal/tensor"
)

// CachedBlockContext: scheduler-owned layer state.
type CachedBlockContext struct {
	Builder          *tensor.Builder
	Input            *tensor.Tensor
	Positions        []uint32
	MultiPositions   *[4][]uint32
	TokenRows        []uint32
	PastKey          *tensor.Tensor
	PastValue        *tensor.Tensor
	PastStates       CacheStates[*tensor.Tensor]
	CurrentPositions *tensor.Tensor
	PerLayerInput    *tensor.Tensor
	Layer            uint32
	Recurrent        bool
}

// BlockDispatchOptions: family-dispatch graph inputs.
type BlockDispatchOptions DenseBlockOptions

// BuildArchitectureBlockCached: family-routed graph construction.
func BuildArchitectureBlockCached(
	options BlockDispatchOptions,
) (DenseBlockResult, error) {
	_, ok := options.Spec.ResolvedProfile()
	if !ok {
		return DenseBlockResult{}, &UnsupportedArchitectureError{
			Architecture: options.Spec.Architecture,
		}
	}
	context := options.Context
	plan := options.Spec.PlanLayer(context.Layer, context.Recurrent)
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
			context.Builder, context.Input, options.Spec, options.Weights,
			context.PastKey, context.PastValue,
		)
	case BlockMamba2:
		return BuildMamba2BlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.PastKey, context.PastValue,
		)
	case BlockFalconH1:
		return BuildFalconH1BlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.Positions, context.PastKey, context.PastValue,
			context.PastStates[CacheStateConvolution].Value,
			context.PastStates[CacheStateSSM].Value,
		)
	case BlockJamba:
		return BuildJambaRecurrentBlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.PastKey, context.PastValue,
		)
	case BlockGraniteHybrid:
		return BuildGraniteHybridRecurrentBlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.PastKey, context.PastValue,
		)
	case BlockPLaMo2:
		return BuildPLaMo2RecurrentBlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.PastKey, context.PastValue,
		)
	case BlockNemotronH:
		return BuildNemotronHBlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.Positions, context.PastKey, context.PastValue, context.Layer,
		)
	case BlockKimiLinear:
		return BuildKimiLinearBlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.Positions, plan.Recurrent, context.PastKey,
			context.PastValue, context.Layer,
		)
	case BlockDSA:
		return BuildDSABlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.Positions, context.PastKey, context.PastValue,
			context.PastStates[CacheStateIndexerKey].Value,
			context.PerLayerInput, context.Layer,
		)
	case BlockDeepSeek4:
		return BuildDeepSeek4BlockCached(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.Positions, context.TokenRows, context.PastKey,
			context.PastStates, context.CurrentPositions, context.Layer,
		)
	case BlockMLA:
		return BuildMLABlockCachedForLayer(
			context.Builder, context.Input, options.Spec, options.Weights,
			context.Positions, context.PastKey, context.PastValue, context.Layer,
		)
	case BlockDense:
		return BuildDenseBlockWithOptions(DenseBlockOptions(options))
	default:
		return DenseBlockResult{}, errors.New("unknown compiled block policy")
	}
}
