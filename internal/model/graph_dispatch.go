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
	return executeLayerProgram(options, plan)
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

type layerExecution struct {
	residual *tensor.Tensor
	current  *tensor.Tensor
	result   DenseBlockResult
}

func executeLayerProgram(
	options BlockDispatchOptions,
	plan LayerPlan,
) (DenseBlockResult, error) {
	if plan.Program.Count == 0 || int(plan.Program.Count) > len(plan.Program.Instructions) {
		return DenseBlockResult{}, errors.New("compiled layer program is invalid")
	}
	execution := layerExecution{
		residual: options.Context.Input, current: options.Context.Input,
	}
	for index := range int(plan.Program.Count) {
		instruction, ok := plan.Program.Instruction(index)
		if !ok {
			return DenseBlockResult{}, errors.New("compiled layer instruction is missing")
		}
		operands, err := resolveLayerOperands(options.Context, instruction)
		if err != nil {
			return DenseBlockResult{}, err
		}
		if err := executeLayerInstruction(options, plan, instruction, operands, &execution); err != nil {
			return DenseBlockResult{}, err
		}
	}
	if execution.result.Output == nil {
		return DenseBlockResult{}, errors.New("compiled layer program produced no output")
	}
	return execution.result, nil
}

func executeLayerInstruction(
	options BlockDispatchOptions,
	plan LayerPlan,
	instruction LayerOperatorInstruction,
	operands layerOperands,
	execution *layerExecution,
) error {
	c := options.Context
	switch instruction.Operator {
	case LayerOperatorAttentionNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			options.Weights.AttentionNorm == nil {
			return errors.New("compiled attention-normalization stage is invalid")
		}
		execution.current = c.Builder.WeightedRMSNorm(
			execution.current, options.Weights.AttentionNorm, options.Spec.RMSNormEpsilon,
		)
		return c.Builder.Err()
	case LayerOperatorRecurrentMix:
		if instruction.CacheCount != 2 {
			return errors.New("compiled recurrent-mixing stage is invalid")
		}
		var result DenseBlockResult
		var err error
		switch instruction.Family {
		case BlockMamba:
			result, err = buildMambaMixerCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1],
			)
		case BlockMamba2:
			if plan.Block == BlockMamba2 && options.Weights.SSMConv1DBias == nil {
				return errors.New("Mamba2 recurrent-mixing convolution bias is nil")
			}
			result, err = buildMamba2MixerCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1],
			)
		default:
			return errors.New("compiled recurrent-mixing policy is invalid")
		}
		if err != nil {
			return err
		}
		execution.result = result
		execution.current = result.Output
		return nil
	case LayerOperatorFeedForwardNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			options.Weights.FeedForwardNorm == nil {
			return errors.New("compiled feed-forward normalization stage is invalid")
		}
		execution.current = c.Builder.WeightedRMSNorm(
			execution.residual, options.Weights.FeedForwardNorm, options.Spec.RMSNormEpsilon,
		)
		return c.Builder.Err()
	case LayerOperatorFeedForwardMix:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 {
			return errors.New("compiled feed-forward stage is invalid")
		}
		feedForward, err := buildStandardFeedForwardMix(
			c.Builder, execution.current, plan, options.Spec, options.Weights,
		)
		if err != nil {
			return err
		}
		execution.current = feedForward
		return nil
	case LayerOperatorScale:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			options.Spec.ResidualScale <= 0 {
			return errors.New("compiled scale stage is invalid")
		}
		execution.current = c.Builder.Scale(execution.current, options.Spec.ResidualScale)
		return c.Builder.Err()
	case LayerOperatorResidual:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			execution.current == nil || execution.residual == nil {
			return errors.New("compiled residual stage is invalid")
		}
		execution.current = c.Builder.Add(execution.residual, execution.current)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case LayerOperatorFamilyBlock:
		result, err := executeFamilyBlock(options, plan, instruction, operands)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	default:
		return errors.New("compiled layer operator is unknown")
	}
}

func executeFamilyBlock(
	options BlockDispatchOptions,
	plan LayerPlan,
	instruction LayerOperatorInstruction,
	operands layerOperands,
) (DenseBlockResult, error) {
	c := options.Context
	if instruction.Family != plan.Block || instruction.Family == BlockMamba ||
		instruction.Family == BlockMamba2 {
		return DenseBlockResult{}, errors.New("compiled family block differs from layer plan")
	}
	switch instruction.Family {
	case BlockDense:
		return BuildDenseBlockWithOptions(DenseBlockOptions(options))
	case BlockFalconH1:
		return BuildFalconH1BlockCached(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions,
			operands.caches[0], operands.caches[1], operands.caches[2], operands.caches[3],
		)
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
		return DenseBlockResult{}, errors.New("compiled family block is unknown")
	}
}

func buildStandardFeedForwardMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	plan LayerPlan,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if weights.FeedForwardRouter != nil {
		required := graphWeights{
			requireGraphWeight("feed-forward router", weights.FeedForwardRouter),
			requireGraphWeight("feed-forward expert down", weights.FeedForwardDownExperts),
		}
		if weights.FeedForwardGateUpExperts != nil {
			required.add("feed-forward fused expert gate/up", weights.FeedForwardGateUpExperts)
		} else if plan.Block != BlockGraniteHybrid {
			required.add("feed-forward expert gate", weights.FeedForwardGateExperts)
			required.add("feed-forward expert up", weights.FeedForwardUpExperts)
		} else {
			required.add("feed-forward expert up", weights.FeedForwardUpExperts)
		}
		if err := required.validate("compiled feed-forward stage"); err != nil {
			return nil, err
		}
		feedForward := plan.Experts.BuildLayer(builder, input, nil, weights)
		if plan.Block == BlockGraniteHybrid && spec.SharedExpertFF > 0 {
			if weights.FeedForwardSharedGate == nil || weights.FeedForwardSharedUp == nil ||
				weights.FeedForwardSharedDown == nil {
				return nil, errors.New("compiled shared feed-forward stage is incomplete")
			}
			feedForward = builder.Add(feedForward, buildSharedSwiGLU(builder, input, weights))
		}
		return feedForward, builder.Err()
	}
	if err := (graphWeights{
		requireGraphWeight("feed-forward gate", weights.FeedForwardGate),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
	}).validate("compiled feed-forward stage"); err != nil {
		return nil, err
	}
	gate := builder.MulMat(weights.FeedForwardGate, input)
	up := builder.MulMat(weights.FeedForwardUp, input)
	if weights.FeedForwardGateBias != nil {
		gate = builder.Add(gate, weights.FeedForwardGateBias)
	}
	if weights.FeedForwardUpBias != nil {
		up = builder.Add(up, weights.FeedForwardUpBias)
	}
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	if weights.FeedForwardDownBias != nil {
		feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
	}
	return feedForward, builder.Err()
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
