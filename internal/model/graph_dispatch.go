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
	Encoder          *tensor.Tensor
	CrossKey         *tensor.Tensor
	CrossValue       *tensor.Tensor
	PastStates       CacheStates[*tensor.Tensor]
	CurrentPositions *tensor.Tensor
	PerLayerInput    *tensor.Tensor
	Layer            uint32
	Recurrent        bool
	CacheWrite       tensor.CacheWriteMode
	Sequences        uint64
}

// BlockDispatchOptions: compiled layer-program inputs.
type BlockDispatchOptions struct {
	Context CachedBlockContext
	Spec    Spec
	Weights LayerGraphWeights
	Plan    *LayerPlan
}

// Build executes a compiled layer program.
func (p CompiledLayerProgram) Build(
	context CachedBlockContext,
	weights LayerGraphWeights,
) (DenseBlockResult, error) {
	plan := p.plan
	return executeCompiledLayer(BlockDispatchOptions{
		Context: context, Spec: p.spec, Weights: weights, Plan: &plan,
	})
}

func executeCompiledLayer(options BlockDispatchOptions) (DenseBlockResult, error) {
	if options.Plan == nil {
		return DenseBlockResult{}, errors.New("compiled layer plan is required")
	}
	plan := *options.Plan
	sequence, _ := plan.Program.Instruction(1)
	if plan.ExplicitEncoder && sequence.Operator != LayerOperatorAttentionRelativeBidirectional &&
		sequence.Operator != LayerOperatorAttentionRelativeCausal {
		return DenseBlockResult{}, errors.New("encoder-decoder blocks require explicit encoder state")
	}
	context := options.Context
	if context.MultiPositions != nil {
		if !plan.MultiAxis {
			return DenseBlockResult{}, errors.New("layer architecture does not support multi-axis positions")
		}
		context.Positions = context.MultiPositions[0]
		options.Context = context
	}
	if plan.Layer != context.Layer || plan.Recurrent != context.Recurrent {
		return DenseBlockResult{}, errors.New("compiled layer plan differs from dispatch context")
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
		case RuntimeCacheCrossKey:
			operands.caches[index] = context.CrossKey
		case RuntimeCacheCrossValue:
			operands.caches[index] = context.CrossValue
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
		case RuntimeTensorEncoder:
			operands.tensors[index] = context.Encoder
		default:
			return operands, errors.New("compiled layer tensor binding is invalid")
		}
	}
	return operands, nil
}

type layerExecution struct {
	residual        *tensor.Tensor
	current         *tensor.Tensor
	attentionInput  *tensor.Tensor
	feedForwardBase *tensor.Tensor
	result          DenseBlockResult
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
	case LayerOperatorAttentionInputNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 {
			return errors.New("compiled attention-input stage is invalid")
		}
		attentionInput, feedForwardBase, err := preparePolicyAttentionInputs(options)
		if err != nil {
			return err
		}
		execution.current = attentionInput
		execution.attentionInput = attentionInput
		execution.feedForwardBase = feedForwardBase
		return nil
	case LayerOperatorAttentionNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			options.Weights.AttentionNorm == nil {
			return errors.New("compiled attention-normalization stage is invalid")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.current, options.Weights.AttentionNorm,
			options.Weights.AttentionNormBias,
		)
		return c.Builder.Err()
	case LayerOperatorCrossAttentionNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			options.Weights.CrossAttentionNorm == nil {
			return errors.New("compiled cross-attention normalization stage is invalid")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.current, options.Weights.CrossAttentionNorm, nil,
		)
		return c.Builder.Err()
	case LayerOperatorRMSNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 || execution.current == nil {
			return errors.New("compiled RMS-normalization stage is invalid")
		}
		execution.current = c.Builder.RMSNorm(execution.current, options.Spec.RMSNormEpsilon)
		return c.Builder.Err()
	case LayerOperatorPairedInputNorm:
		target := operands.tensors[0]
		if instruction.CacheCount != 0 || instruction.TensorCount != 1 || target == nil ||
			execution.current == nil || !execution.current.Shape.Equal(target.Shape) ||
			execution.current.Shape.Rank != 2 ||
			execution.current.Shape.Dims[0] != uint64(options.Spec.EmbeddingLength) ||
			options.Weights.AttentionNorm == nil || options.Weights.AttentionNorm2 == nil {
			return errors.New("compiled paired-input normalization stage is invalid")
		}
		tokenNorm := c.Builder.WeightedRMSNorm(
			execution.current, options.Weights.AttentionNorm, options.Spec.RMSNormEpsilon,
		)
		targetNorm := c.Builder.WeightedRMSNorm(
			target, options.Weights.AttentionNorm2, options.Spec.RMSNormEpsilon,
		)
		execution.current = c.Builder.Concat(tokenNorm, targetNorm, 0)
		execution.residual = target
		if options.Spec.NormBeforeResidual {
			execution.residual = targetNorm
		}
		return c.Builder.Err()
	case LayerOperatorAttentionPostNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			options.Weights.AttentionPostNorm == nil {
			return errors.New("compiled attention post-normalization stage is invalid")
		}
		execution.current = c.Builder.WeightedRMSNorm(
			execution.current, options.Weights.AttentionPostNorm, options.Spec.RMSNormEpsilon,
		)
		return c.Builder.Err()
	case LayerOperatorAttentionCausalProjection, LayerOperatorAttentionGatedProjection,
		LayerOperatorAttentionOutputProjection, LayerOperatorAttentionBidirectionalFusedQKV,
		LayerOperatorAttentionBidirectionalQKNorm, LayerOperatorAttentionCausalPostQKNorm,
		LayerOperatorAttentionSharedKVQKNorm, LayerOperatorAttentionBidirectionalEncoder,
		LayerOperatorAttentionPairedCausalProjection, LayerOperatorAttentionSharedCacheQKNorm,
		LayerOperatorAttentionPlannedProjection, LayerOperatorAttentionRelativeBidirectional,
		LayerOperatorAttentionRelativeCausal, LayerOperatorAttentionCross:
		cacheCount, tensorCount := uint8(2), uint8(0)
		switch instruction.Operator {
		case LayerOperatorAttentionOutputProjection, LayerOperatorAttentionRelativeBidirectional:
			cacheCount = 0
		case LayerOperatorAttentionCross:
			tensorCount = 1
		}
		if instruction.TensorCount != tensorCount || instruction.CacheCount != cacheCount {
			return errors.New("compiled attention-mixing stage is invalid")
		}
		var result DenseBlockResult
		var err error
		switch instruction.Operator {
		case LayerOperatorAttentionCausalProjection:
			result, err = buildCausalProjectionMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan.Layer,
			)
		case LayerOperatorAttentionGatedProjection:
			sequences := c.Sequences
			if sequences == 0 {
				sequences = 1
			}
			result, err = buildGatedProjectionMixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				c.Positions, c.MultiPositions, sequences,
				operands.caches[0], operands.caches[1], c.CacheWrite,
				plan.AttentionGraph.deltaProjection,
			)
		case LayerOperatorAttentionOutputProjection:
			if options.Weights.AttentionOutput == nil {
				return errors.New("compiled output-projection stage is invalid")
			}
			projected := c.Builder.MulMat(options.Weights.AttentionOutput, execution.current)
			if options.Weights.AttentionOutputBias != nil {
				projected = c.Builder.Add(projected, options.Weights.AttentionOutputBias)
			}
			result.Output, err = projected, c.Builder.Err()
		case LayerOperatorAttentionBidirectionalFusedQKV:
			result, err = buildBidirectionalFusedQKVMix(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan,
			)
		case LayerOperatorAttentionBidirectionalQKNorm:
			result, err = buildBidirectionalQKNormMix(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan,
			)
		case LayerOperatorAttentionCausalPostQKNorm:
			result, err = buildCausalPostQKNormMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan.Layer,
			)
		case LayerOperatorAttentionSharedKVQKNorm:
			result, err = buildSharedKVQKNormMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan, c.CacheWrite,
			)
		case LayerOperatorAttentionBidirectionalEncoder:
			result, err = buildBidirectionalEncoderAttentionMix(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan,
			)
		case LayerOperatorAttentionPairedCausalProjection:
			result, err = buildPairedCausalProjectionMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], c.CacheWrite,
			)
		case LayerOperatorAttentionSharedCacheQKNorm:
			result, err = buildSharedCacheQKNormMix(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan,
			)
		case LayerOperatorAttentionPlannedProjection:
			result, err = buildPolicyAttentionMix(
				options, execution.current, execution.feedForwardBase,
				operands.caches[0], operands.caches[1],
			)
		case LayerOperatorAttentionRelativeBidirectional:
			result, err = buildRelativeSelfAttentionMix(
				c.Builder, execution.current, options.Spec, options.Weights, nil, nil, false,
			)
		case LayerOperatorAttentionRelativeCausal:
			result, err = buildRelativeSelfAttentionMix(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1], true,
			)
		case LayerOperatorAttentionCross:
			result, err = buildCrossAttentionMix(
				c.Builder, execution.current, operands.tensors[0], options.Spec, options.Weights,
				operands.caches[0], operands.caches[1],
			)
		default:
			return errors.New("compiled attention-mixing policy is invalid")
		}
		if err != nil {
			return err
		}
		execution.result.Output = result.Output
		if result.Key != nil || result.Value != nil {
			execution.result.Key, execution.result.Value = result.Key, result.Value
		}
		// Preserve optional exact-attention inputs.
		if result.Query != nil {
			execution.result.Query, execution.result.AttentionScale = result.Query, result.AttentionScale
		}
		if len(result.States) > 0 {
			execution.result.States = result.States
		}
		execution.current = result.Output
		return nil
	case LayerOperatorHybridMix:
		if instruction.CacheCount != 4 || instruction.TensorCount != 0 {
			return errors.New("compiled hybrid-mixing stage is invalid")
		}
		result, err := buildAttentionSSMHybridMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[0], operands.caches[1], operands.caches[2], operands.caches[3], plan,
		)
		if err != nil {
			return err
		}
		execution.result = result
		execution.current = result.Output
		return nil
	case LayerOperatorRecurrentMix:
		if instruction.CacheCount != 2 {
			return errors.New("compiled recurrent-mixing stage is invalid")
		}
		var result DenseBlockResult
		var err error
		switch plan.Mixer {
		case recurrentMixerSelectiveScan, recurrentMixerWeightedSelectiveScan:
			result, err = buildSelectiveScanMixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1], plan.Mixer,
			)
		case recurrentMixerGroupedSelectiveScan, recurrentMixerScaledGroupedSelectiveScan,
			recurrentMixerSparseGroupedSelectiveScan:
			if plan.Mixer == recurrentMixerGroupedSelectiveScan &&
				options.Weights.SSMConv1DBias == nil {
				return errors.New("grouped selective-scan convolution bias is nil")
			}
			result, err = buildGroupedSelectiveScanMixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1], plan.Mixer,
			)
		case recurrentMixerNormalizedSelectiveScan:
			result, err = buildNormalizedSelectiveScanMixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1],
			)
		case recurrentMixerGatedDelta:
			sequences := c.Sequences
			if sequences == 0 {
				sequences = 1
			}
			result, err = buildGatedDeltaMixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				c.Positions, sequences, operands.caches[0], operands.caches[1],
				plan.AttentionGraph.deltaProjection,
			)
		case recurrentMixerShortConvolution:
			result, err = buildShortConvolutionMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1],
			)
		case recurrentMixerDynamicWKV6:
			result, err = buildDynamicWKV6MixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1],
			)
		case recurrentMixerAffineWKV6:
			result, err = buildAffineWKV6MixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1], plan,
			)
		case recurrentMixerDynamicWKV7:
			result, err = buildDynamicWKV7MixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1], plan,
			)
		case recurrentMixerKeyedDelta:
			result, err = buildKeyedDeltaAttentionMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
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
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.residual, options.Weights.FeedForwardNorm,
			options.Weights.FeedForwardNormBias,
		)
		return c.Builder.Err()
	case LayerOperatorFeedForwardInputNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			execution.attentionInput == nil || execution.feedForwardBase == nil {
			return errors.New("compiled feed-forward input stage is invalid")
		}
		normalized := plan.ResidualStages.AfterAttention(
			c.Builder, execution.residual, execution.attentionInput, options.Spec, options.Weights,
		)
		if plan.ExpertComposition.kind != expertDenseRoutedSeparateNorm {
			normalized = plan.ResidualStages.FeedForwardInput(
				c.Builder, c.Input, execution.residual, normalized, execution.feedForwardBase,
				options.Spec, options.Weights, plan.Normalization,
			)
		}
		execution.current = normalized
		return c.Builder.Err()
	case LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorFeedForwardFusedGLU,
		LayerOperatorFeedForwardSquaredReLU, LayerOperatorFeedForwardRoutedSquaredReLU,
		LayerOperatorFeedForwardRoutedSwiGLU, LayerOperatorFeedForwardGatedGELU,
		LayerOperatorFeedForwardParallelGatedGELU, LayerOperatorFeedForwardEncoder,
		LayerOperatorFeedForwardPlanned, LayerOperatorFeedForwardRelative:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 {
			return errors.New("compiled feed-forward stage is invalid")
		}
		var feedForward *tensor.Tensor
		var err error
		switch instruction.Operator {
		case LayerOperatorFeedForwardStandardSwiGLU:
			feedForward, err = buildStandardFeedForwardMix(
				c.Builder, execution.current, plan, options.Spec, options.Weights,
			)
		case LayerOperatorFeedForwardFusedGLU:
			feedForward, err = buildFusedFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights, plan.Layer,
			)
		case LayerOperatorFeedForwardSquaredReLU:
			feedForward, err = buildSquaredReLUFeedForwardMix(
				c.Builder, execution.current, options.Weights,
			)
		case LayerOperatorFeedForwardRoutedSquaredReLU:
			feedForward, err = buildRoutedSquaredReLUFeedForwardMix(
				c.Builder, execution.current, options.Weights, plan.Experts,
			)
		case LayerOperatorFeedForwardRoutedSwiGLU:
			feedForward, err = buildRoutedSwiGLUFeedForwardMix(
				c.Builder, execution.current, options.Weights,
				plan.Experts, plan.ExpertComposition,
			)
		case LayerOperatorFeedForwardGatedGELU:
			feedForward, err = buildGatedGELUFeedForwardMix(
				c.Builder, execution.current, options.Weights,
			)
		case LayerOperatorFeedForwardParallelGatedGELU:
			feedForward, err = buildParallelGatedGELUFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights, plan,
			)
		case LayerOperatorFeedForwardEncoder:
			feedForward, err = buildEncoderFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights, plan,
			)
		case LayerOperatorFeedForwardPlanned:
			feedForward, err = buildPolicyFeedForwardMix(
				options, execution.current, execution.residual,
			)
		case LayerOperatorFeedForwardRelative:
			feedForward, err = buildRelativeFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights,
			)
		default:
			return errors.New("compiled feed-forward policy is invalid")
		}
		if err != nil {
			return err
		}
		execution.current = feedForward
		return nil
	case LayerOperatorFeedForwardOutput:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 || execution.current == nil {
			return errors.New("compiled feed-forward output stage is invalid")
		}
		if options.Weights.FeedForwardRouter == nil {
			execution.current = plan.ResidualStages.ApplyFeedForwardOutput(
				c.Builder, execution.current, options.Spec, options.Weights, plan.Normalization,
			)
		}
		return c.Builder.Err()
	case LayerOperatorFeedForwardPostNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			options.Weights.FeedForwardPostNorm == nil {
			return errors.New("compiled feed-forward post-normalization stage is invalid")
		}
		execution.current = c.Builder.WeightedRMSNorm(
			execution.current, options.Weights.FeedForwardPostNorm, options.Spec.RMSNormEpsilon,
		)
		return c.Builder.Err()
	case LayerOperatorCacheSentinel:
		if instruction.CacheCount != 2 || instruction.TensorCount != 0 {
			return errors.New("compiled sentinel-cache stage is invalid")
		}
		if len(c.Positions) == 0 || uint64(len(c.Positions)) != execution.residual.Shape.Dims[1] {
			return errors.New("compiled sentinel-cache positions are invalid")
		}
		key, value, err := buildSentinelCache(
			c.Builder, execution.residual, operands.caches[0], operands.caches[1],
		)
		if err != nil {
			return err
		}
		execution.result.Key = key
		execution.result.Value = value
		execution.result.Output = execution.residual
		return nil
	case LayerOperatorScale:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			plan.ResidualStages.residualScale <= 0 {
			return errors.New("compiled scale stage is invalid")
		}
		execution.current = c.Builder.Scale(execution.current, plan.ResidualStages.residualScale)
		execution.result.Output = execution.current
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
	case LayerOperatorResidualScale:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			execution.current == nil || execution.residual == nil || options.Weights.LayerOutputScale == nil {
			return errors.New("compiled residual-scale stage is invalid")
		}
		execution.current = c.Builder.Multiply(
			c.Builder.Add(execution.residual, execution.current), options.Weights.LayerOutputScale,
		)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case LayerOperatorAttentionResidualNorm, LayerOperatorInputResidualNorm,
		LayerOperatorFeedForwardResidualNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			execution.current == nil || execution.residual == nil {
			return errors.New("compiled residual-normalization stage is invalid")
		}
		residual, weight, bias := execution.residual, options.Weights.AttentionPostNorm,
			options.Weights.AttentionPostNormBias
		switch instruction.Operator {
		case LayerOperatorInputResidualNorm:
			residual, weight, bias = c.Input, options.Weights.AttentionNorm2, options.Weights.AttentionNorm2Bias
		case LayerOperatorFeedForwardResidualNorm:
			weight, bias = options.Weights.FeedForwardPostNorm, options.Weights.FeedForwardPostNormBias
		}
		if instruction.Operator == LayerOperatorInputResidualNorm && weight == nil && bias == nil {
			return nil
		}
		if weight == nil || bias == nil {
			return errors.New("compiled residual-normalization weights are incomplete")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, c.Builder.Add(residual, execution.current), weight, bias,
		)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case LayerOperatorScaledSkip:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 || execution.current == nil ||
			options.Weights.EmbeddingSkip == nil || options.Weights.LayerOutputScale == nil ||
			!options.Weights.EmbeddingSkip.Shape.Equal(execution.current.Shape) {
			return errors.New("compiled scaled-skip stage is invalid")
		}
		execution.current = c.Builder.Add(
			execution.current,
			c.Builder.Multiply(options.Weights.EmbeddingSkip, options.Weights.LayerOutputScale),
		)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case LayerOperatorOutputAdapter:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 || execution.current == nil {
			return errors.New("compiled output-adapter stage is invalid")
		}
		output, err := buildOutputAdapter(c.Builder, execution.current, options.Spec, options.Weights, plan)
		if err != nil {
			return err
		}
		execution.current = output
		execution.residual = output
		execution.result.Output = output
		return nil
	case LayerOperatorGatedTokenShiftSquaredReLU, LayerOperatorTokenShiftSquaredReLU:
		if instruction.CacheCount != 1 || instruction.TensorCount != 0 || execution.current == nil {
			return errors.New("compiled token-shift stage is invalid")
		}
		result, err := buildTokenShiftFeedForwardMix(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[0], execution.result.Key, instruction.Operator, plan.Normalization,
			plan.RecurrentRuntime.TokenShiftCount,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result.Key = result.Key
		return nil
	case LayerOperatorPeriodicScale:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 || execution.current == nil ||
			plan.PeriodicScale <= 0 {
			return errors.New("compiled periodic-scale stage is invalid")
		}
		execution.current = c.Builder.Scale(execution.current, plan.PeriodicScale)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case LayerOperatorLatentAttention:
		valid := instruction.CacheCount == 2 && instruction.TensorCount == 0 ||
			instruction.CacheCount == 3 && instruction.TensorCount == 1
		if !valid {
			return errors.New("compiled latent-attention stage is invalid")
		}
		result, err := buildLatentAttentionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[0], operands.caches[1], operands.caches[2], operands.tensors[0], plan,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	case LayerOperatorHyperAttention:
		if instruction.CacheCount != 1 || instruction.TensorCount != 1 {
			return errors.New("compiled hyper-attention stage is invalid")
		}
		result, err := buildHyperAttentionStage(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions,
			operands.caches[0], c.PastStates, operands.tensors[0], plan,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	case LayerOperatorHyperFeedForward:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 {
			return errors.New("compiled hyper-feed-forward stage is invalid")
		}
		output, err := buildHyperFeedForwardStage(
			c.Builder, execution.current, options.Spec, options.Weights, c.TokenRows, plan,
		)
		if err != nil {
			return err
		}
		execution.current = output
		execution.result.Output = output
		return nil
	default:
		return errors.New("compiled layer operator is unknown")
	}
}

func buildSentinelCache(
	builder *tensor.Builder,
	input, pastKey, pastValue *tensor.Tensor,
) (*tensor.Tensor, *tensor.Tensor, error) {
	const (
		sentinelOffset      = 0
		sentinelWidth       = 1
		sentinelGroupCount  = 1
		cacheTokenDimension = 2
	)
	if input == nil || input.Shape.Rank != 2 {
		return nil, nil, errors.New("sentinel-cache input is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "sentinel cache must contain both tensors"); err != nil {
		return nil, nil, err
	}
	// distinct node per cache stream: retained-output indexing rejects the
	// same tensor appearing as both Key and Value
	sentinel := func() *tensor.Tensor {
		return builder.GroupSlice(
			input, sentinelOffset, sentinelWidth, sentinelGroupCount, input.Shape.Dims[0],
		)
	}
	cacheKey, cacheValue := sentinel(), sentinel()
	if pastKey != nil {
		wantPrefix, err := tensor.NewShape(
			sentinelWidth, sentinelGroupCount, pastKey.Shape.Dims[cacheTokenDimension],
		)
		if err != nil || !pastKey.Shape.Equal(wantPrefix) || !pastValue.Shape.Equal(wantPrefix) {
			return nil, nil, errors.New("sentinel cache shape is invalid")
		}
		cacheKey = builder.Concat(pastKey, cacheKey, cacheTokenDimension)
		cacheValue = builder.Concat(pastValue, cacheValue, cacheTokenDimension)
	}
	return cacheKey, cacheValue, builder.Err()
}

func buildFusedFeedForwardMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	layer uint32,
) (*tensor.Tensor, error) {
	if err := (graphWeights{
		requireGraphWeight("feed-forward fused gate/up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
	}).validate("compiled fused feed-forward stage"); err != nil {
		return nil, err
	}
	width := uint64(spec.LayerFeedForwardLength(layer))
	tokens := input.Shape.Dims[1]
	fusedWidth := 2 * width
	fused := builder.MulMat(weights.FeedForwardUp, input)
	gate := builder.Reshape(builder.GroupSlice(fused, 0, width, 1, fusedWidth), width, tokens)
	up := builder.Reshape(builder.GroupSlice(fused, width, width, 1, fusedWidth), width, tokens)
	var activated *tensor.Tensor
	switch spec.HiddenActivation {
	case "reglu":
		activated = builder.ReGLU(gate, up)
	case "gelu", "geglu":
		activated = builder.GEGLU(gate, up)
	default:
		activated = builder.SwiGLU(gate, up)
	}
	return builder.MulMat(weights.FeedForwardDown, activated), builder.Err()
}

func buildSquaredReLUFeedForwardMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if err := (graphWeights{
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
	}).validate("compiled squared-ReLU feed-forward stage"); err != nil {
		return nil, err
	}
	up := builder.MulMat(weights.FeedForwardUp, input)
	return builder.MulMat(weights.FeedForwardDown, builder.ReLUSquared(up)), builder.Err()
}

func buildGatedGELUFeedForwardMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if err := (graphWeights{
		requireGraphWeight("feed-forward gate", weights.FeedForwardGate),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
	}).validate("compiled gated-GELU feed-forward stage"); err != nil {
		return nil, err
	}
	gate := builder.MulMat(weights.FeedForwardGate, input)
	up := builder.MulMat(weights.FeedForwardUp, input)
	return builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up)), builder.Err()
}

func buildStandardFeedForwardMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	plan LayerPlan,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if weights.FeedForwardRouter != nil {
		if plan.Experts.OptionalSelectionBias {
			plan.Experts.SelectionBias = weights.FeedForwardExpertBias != nil
		}
		required := graphWeights{
			requireGraphWeight("feed-forward router", weights.FeedForwardRouter),
			requireGraphWeight("feed-forward expert down", weights.FeedForwardDownExperts),
		}
		if weights.FeedForwardGateUpExperts != nil {
			required.add("feed-forward fused expert gate/up", weights.FeedForwardGateUpExperts)
		} else {
			required.add("feed-forward expert up", weights.FeedForwardUpExperts)
			if !plan.DenseWeights.allowUngatedExperts || weights.FeedForwardGateExperts != nil {
				required.add("feed-forward expert gate", weights.FeedForwardGateExperts)
			}
		}
		if plan.Experts.SelectionBias {
			required.add("feed-forward selection bias", weights.FeedForwardExpertBias)
		}
		switch plan.ExpertComposition.kind {
		case expertSharedAdd, expertSharedAverage, expertSharedLimited:
			required.add("shared expert gate", weights.FeedForwardSharedGate)
			required.add("shared expert up", weights.FeedForwardSharedUp)
			required.add("shared expert down", weights.FeedForwardSharedDown)
		case expertSharedGated:
			required.add("shared expert router", weights.FeedForwardSharedRouter)
			required.add("shared expert gate", weights.FeedForwardSharedGate)
			required.add("shared expert up", weights.FeedForwardSharedUp)
			required.add("shared expert down", weights.FeedForwardSharedDown)
		}
		if err := required.validate("compiled feed-forward stage"); err != nil {
			return nil, err
		}
		return plan.ExpertComposition.Build(builder, input, input, input, spec, weights, plan)
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
