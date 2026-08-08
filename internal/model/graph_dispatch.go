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
	instruction, ok := plan.Program.Instruction(0)
	if !ok || plan.Program.Count != 1 || instruction.Operator != plan.Block {
		return DenseBlockResult{}, errors.New("compiled layer operator is invalid")
	}
	operands, err := resolveLayerOperands(context, instruction)
	if err != nil {
		return DenseBlockResult{}, err
	}
	return executeLayerOperator(options, plan, instruction, operands)
}

type layerOperands struct {
	caches  [maxLayerCacheBindings]*tensor.Tensor
	tensors [maxLayerTensorBindings]*tensor.Tensor
}

func resolveLayerOperands(
	context CachedBlockContext,
	instruction LayerOperatorInstruction,
) (layerOperands, error) {
	var operands layerOperands
	if int(instruction.CacheCount) > len(operands.caches) ||
		int(instruction.TensorCount) > len(operands.tensors) {
		return operands, errors.New("compiled layer operand count is invalid")
	}
	for index := range int(instruction.CacheCount) {
		switch instruction.Caches[index] {
		case RuntimeCachePrimaryKey:
			operands.caches[index] = context.PastKey
		case RuntimeCachePrimaryValue:
			operands.caches[index] = context.PastValue
		case RuntimeCacheConvolution:
			operands.caches[index] = context.PastStates[CacheStateConvolution].Value
		case RuntimeCacheSSM:
			operands.caches[index] = context.PastStates[CacheStateSSM].Value
		case RuntimeCacheIndexerKey:
			operands.caches[index] = context.PastStates[CacheStateIndexerKey].Value
		default:
			return operands, errors.New("compiled layer cache binding is invalid")
		}
	}
	for index := range int(instruction.TensorCount) {
		switch instruction.Tensors[index] {
		case RuntimeTensorPerLayerInput:
			operands.tensors[index] = context.PerLayerInput
		case RuntimeTensorCurrentPositions:
			operands.tensors[index] = context.CurrentPositions
		default:
			return operands, errors.New("compiled layer tensor binding is invalid")
		}
	}
	return operands, nil
}

func executeLayerOperator(
	options BlockDispatchOptions,
	plan LayerPlan,
	instruction LayerOperatorInstruction,
	operands layerOperands,
) (DenseBlockResult, error) {
	c := options.Context
	switch instruction.Operator {
	case BlockDense:
		return BuildDenseBlockWithOptions(DenseBlockOptions(options))
	case BlockMamba:
		return BuildMambaBlockCached(c.Builder, c.Input, options.Spec, options.Weights, operands.caches[0], operands.caches[1])
	case BlockMamba2:
		return BuildMamba2BlockCached(c.Builder, c.Input, options.Spec, options.Weights, operands.caches[0], operands.caches[1])
	case BlockFalconH1:
		return BuildFalconH1BlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions,
			operands.caches[0], operands.caches[1], operands.caches[2], operands.caches[3],
		)
	case BlockJamba:
		return BuildJambaRecurrentBlockCached(c.Builder, c.Input, options.Spec, options.Weights, operands.caches[0], operands.caches[1])
	case BlockGraniteHybrid:
		return BuildGraniteHybridRecurrentBlockCached(c.Builder, c.Input, options.Spec, options.Weights, operands.caches[0], operands.caches[1])
	case BlockPLaMo2:
		return BuildPLaMo2RecurrentBlockCached(c.Builder, c.Input, options.Spec, options.Weights, operands.caches[0], operands.caches[1])
	case BlockNemotronH:
		return BuildNemotronHBlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions,
			operands.caches[0], operands.caches[1], c.Layer,
		)
	case BlockKimiLinear:
		return BuildKimiLinearBlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions, plan.Recurrent,
			operands.caches[0], operands.caches[1], c.Layer,
		)
	case BlockMLA:
		return BuildMLABlockCachedForLayer(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions,
			operands.caches[0], operands.caches[1], c.Layer,
		)
	case BlockDSA:
		return BuildDSABlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions,
			operands.caches[0], operands.caches[1], operands.caches[2], operands.tensors[0], c.Layer,
		)
	case BlockDeepSeek4:
		return BuildDeepSeek4BlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions, c.TokenRows,
			operands.caches[0], c.PastStates, operands.tensors[0], c.Layer,
		)
	case BlockQwenGDN:
		return executeQwenGDNOperator(options, plan, operands)
	default:
		return DenseBlockResult{}, errors.New("compiled layer operator is unknown")
	}
}

func executeQwenGDNOperator(
	options BlockDispatchOptions,
	plan LayerPlan,
	operands layerOperands,
) (DenseBlockResult, error) {
	c := options.Context
	sequences := c.Sequences
	if sequences == 0 {
		sequences = 1
	}
	pastKey, pastValue := operands.caches[0], operands.caches[1]
	convState, ssmState := operands.caches[2], operands.caches[3]
	if plan.Recurrent {
		convState, ssmState = operands.caches[0], operands.caches[1]
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
