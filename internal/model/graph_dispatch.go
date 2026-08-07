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
	CacheWrite       tensor.CacheWriteMode
	Sequences        uint64
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
	if options.Plan == nil {
		return DenseBlockResult{}, errors.New("compiled layer plan is required")
	}
	context := options.Context
	plan := *options.Plan
	if plan.Layer != context.Layer || plan.Recurrent != context.Recurrent {
		return DenseBlockResult{}, errors.New("compiled layer plan differs from dispatch context")
	}
	if plan.GraphFamily == ArchitectureFamilyEncoderDecoder {
		return DenseBlockResult{}, errors.New(
			"encoder-decoder blocks require explicit encoder state",
		)
	}
	if int(plan.Block) >= len(cachedBlockCatalog) || cachedBlockCatalog[plan.Block] == nil {
		return DenseBlockResult{}, errors.New("unknown compiled block policy")
	}
	return cachedBlockCatalog[plan.Block](options, plan)
}

type cachedBlockBuilder func(BlockDispatchOptions, LayerPlan) (DenseBlockResult, error)

var cachedBlockCatalog = [...]cachedBlockBuilder{
	BlockDense: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		return BuildDenseBlockWithOptions(DenseBlockOptions(options))
	},
	BlockMamba: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildMambaBlockCached(c.Builder, c.Input, options.Spec, options.Weights, c.PastKey, c.PastValue)
	},
	BlockMamba2: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildMamba2BlockCached(c.Builder, c.Input, options.Spec, options.Weights, c.PastKey, c.PastValue)
	},
	BlockFalconH1: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildFalconH1BlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions, c.PastKey, c.PastValue,
			c.PastStates[CacheStateConvolution].Value, c.PastStates[CacheStateSSM].Value,
		)
	},
	BlockJamba: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildJambaRecurrentBlockCached(c.Builder, c.Input, options.Spec, options.Weights, c.PastKey, c.PastValue)
	},
	BlockGraniteHybrid: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildGraniteHybridRecurrentBlockCached(c.Builder, c.Input, options.Spec, options.Weights, c.PastKey, c.PastValue)
	},
	BlockPLaMo2: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildPLaMo2RecurrentBlockCached(c.Builder, c.Input, options.Spec, options.Weights, c.PastKey, c.PastValue)
	},
	BlockNemotronH: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildNemotronHBlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions, c.PastKey, c.PastValue, c.Layer,
		)
	},
	BlockKimiLinear: func(options BlockDispatchOptions, plan LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildKimiLinearBlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions, plan.Recurrent,
			c.PastKey, c.PastValue, c.Layer,
		)
	},
	BlockMLA: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildMLABlockCachedForLayer(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions, c.PastKey, c.PastValue, c.Layer,
		)
	},
	BlockDSA: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildDSABlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions, c.PastKey, c.PastValue,
			c.PastStates[CacheStateIndexerKey].Value, c.PerLayerInput, c.Layer,
		)
	},
	BlockDeepSeek4: func(options BlockDispatchOptions, _ LayerPlan) (DenseBlockResult, error) {
		c := options.Context
		return BuildDeepSeek4BlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions, c.TokenRows,
			c.PastKey, c.PastStates, c.CurrentPositions, c.Layer,
		)
	},
	BlockQwenGDN: buildQwenGDNBlockCached,
}

func buildQwenGDNBlockCached(options BlockDispatchOptions, plan LayerPlan) (DenseBlockResult, error) {
	c := options.Context
	sequences := c.Sequences
	if sequences == 0 {
		sequences = 1
	}
	pastKey, pastValue := c.PastKey, c.PastValue
	convState := c.PastStates[CacheStateConvolution].Value
	ssmState := c.PastStates[CacheStateSSM].Value
	if plan.Recurrent {
		convState, ssmState = pastKey, pastValue
		pastKey, pastValue = nil, nil
	}
	result, err := BuildQwen35BlockWithOptions(Qwen35BlockOptions{
		Builder: c.Builder, Input: c.Input, Spec: options.Spec, Weights: options.Weights,
		Positions: c.Positions, MultiPositions: c.MultiPositions, Sequences: sequences,
		Recurrent: plan.Recurrent, PastKey: pastKey, PastValue: pastValue,
		ConvState: convState, SSMState: ssmState, CacheWrite: c.CacheWrite,
	})
	if err != nil {
		return DenseBlockResult{}, err
	}
	if plan.Recurrent {
		return DenseBlockResult{Output: result.Output, Key: result.ConvState, Value: result.SSMState}, nil
	}
	return DenseBlockResult{Output: result.Output, Key: result.Key, Value: result.Value}, nil
}
