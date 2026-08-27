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
	MultiPositions   *[tensor.MaxDimensions][]uint32
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

type blockDispatchOptions struct {
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
	return executeCompiledLayer(blockDispatchOptions{
		Context: context, Spec: p.spec, Weights: weights, Plan: &plan,
	})
}

func executeCompiledLayer(options blockDispatchOptions) (DenseBlockResult, error) {
	if options.Plan == nil {
		return DenseBlockResult{}, errors.New("compiled layer plan is required")
	}
	plan := *options.Plan
	context := options.Context
	if context.Sequences == tensor.FirstOffset {
		context.Sequences = tensor.SingletonExtent
		options.Context = context
	}
	if context.MultiPositions != nil {
		if !plan.MultiAxis {
			return DenseBlockResult{}, errors.New("layer architecture does not support multi-axis positions")
		}
		context.Positions = context.MultiPositions[tensor.FirstOffset]
		options.Context = context
	}
	if plan.Layer != context.Layer || plan.Recurrent != context.Recurrent {
		return DenseBlockResult{}, errors.New("compiled layer plan differs from dispatch context")
	}
	return executeLayerProgram(options, plan)
}

type layerOperands struct {
	caches  [runtimeCacheBindingCount]*tensor.Tensor
	tensors [runtimeTensorBindingCount]*tensor.Tensor
}

func resolveLayerOperands(
	context CachedBlockContext,
	instruction LayerOperatorInstruction,
) layerOperands {
	var operands layerOperands
	for index := range instruction.Caches[:instruction.CacheCount] {
		binding := instruction.Caches[index]
		switch binding {
		case RuntimeCachePrimaryKey:
			operands.caches[binding] = context.PastKey
		case RuntimeCachePrimaryValue:
			operands.caches[binding] = context.PastValue
		case RuntimeCacheConvolution:
			operands.caches[binding] = context.PastStates[CacheStateConvolution].Value
		case RuntimeCacheSSM:
			operands.caches[binding] = context.PastStates[CacheStateSSM].Value
		case RuntimeCacheIndexerKey:
			operands.caches[binding] = context.PastStates[CacheStateIndexerKey].Value
		case RuntimeCacheCrossKey:
			operands.caches[binding] = context.CrossKey
		case RuntimeCacheCrossValue:
			operands.caches[binding] = context.CrossValue
		}
	}
	for index := range instruction.Tensors[:instruction.TensorCount] {
		binding := instruction.Tensors[index]
		switch binding {
		case RuntimeTensorPerLayerInput:
			operands.tensors[binding] = context.PerLayerInput
		case RuntimeTensorCurrentPositions:
			operands.tensors[binding] = context.CurrentPositions
		case RuntimeTensorEncoder:
			operands.tensors[binding] = context.Encoder
		}
	}
	return operands
}

type layerExecution struct {
	residual        *tensor.Tensor
	current         *tensor.Tensor
	attentionInput  *tensor.Tensor
	feedForwardBase *tensor.Tensor
	result          DenseBlockResult
}

func mergeLayerResult(destination *DenseBlockResult, result DenseBlockResult) {
	destination.Output = result.Output
	if result.Key != nil || result.Value != nil {
		destination.Key, destination.Value = result.Key, result.Value
	}
	if result.Auxiliary != nil {
		destination.Auxiliary = result.Auxiliary
	}
	if result.Query != nil {
		destination.Query, destination.AttentionScale = result.Query, result.AttentionScale
	}
	if len(result.States) > tensor.FirstOffset {
		destination.States = result.States
	}
}

func executeAttentionOperator(
	options blockDispatchOptions,
	plan LayerPlan,
	operator LayerOperator,
	operands layerOperands,
	execution *layerExecution,
) (DenseBlockResult, error) {
	c := options.Context
	switch operator {
	case LayerOperatorAttentionCausalProjection:
		return buildCausalProjectionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan.Layer,
		)
	case LayerOperatorAttentionGatedProjection:
		return buildGatedProjectionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			c.Positions, c.MultiPositions, c.Sequences,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], c.CacheWrite,
			plan.AttentionGraph.deltaProjection,
		)
	case LayerOperatorAttentionOutputProjection:
		if options.Weights.AttentionOutput == nil {
			return DenseBlockResult{}, errors.New("compiled output-projection stage is invalid")
		}
		projected := c.Builder.MulMat(options.Weights.AttentionOutput, execution.current)
		if options.Weights.AttentionOutputBias != nil {
			projected = c.Builder.Add(projected, options.Weights.AttentionOutputBias)
		}
		return DenseBlockResult{Output: projected}, c.Builder.Err()
	case LayerOperatorAttentionBidirectionalFusedQKV:
		return buildBidirectionalFusedQKVMix(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan,
		)
	case LayerOperatorAttentionBidirectionalQKNorm:
		return buildBidirectionalQKNormMix(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan,
		)
	case LayerOperatorAttentionCausalPostQKNorm:
		return buildCausalPostQKNormMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan.Layer,
		)
	case LayerOperatorAttentionSharedKVQKNorm:
		return buildSharedKVQKNormMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan, c.CacheWrite,
		)
	case LayerOperatorAttentionBidirectionalEncoder:
		return buildBidirectionalEncoderAttentionMix(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan,
		)
	case LayerOperatorAttentionPairedCausalProjection:
		return buildPairedCausalProjectionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], c.CacheWrite,
		)
	case LayerOperatorAttentionSharedCacheQKNorm:
		return buildSharedCacheQKNormMix(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan,
		)
	case LayerOperatorAttentionPlannedProjection:
		return buildPolicyAttentionMix(
			options, execution.current, execution.feedForwardBase,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue],
		)
	case LayerOperatorAttentionRelativeBidirectional:
		return buildRelativeSelfAttentionMix(
			c.Builder, execution.current, options.Spec, options.Weights, nil, nil, false,
		)
	case LayerOperatorAttentionRelativeCausal:
		return buildRelativeSelfAttentionMix(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], true,
		)
	case LayerOperatorAttentionCross:
		return buildCrossAttentionMix(
			c.Builder, execution.current, operands.tensors[RuntimeTensorEncoder], options.Spec, options.Weights,
			operands.caches[RuntimeCacheCrossKey], operands.caches[RuntimeCacheCrossValue],
		)
	default:
		return DenseBlockResult{}, errors.New("compiled attention-mixing policy is invalid")
	}
}

func executeRecurrentOperator(
	options blockDispatchOptions,
	plan LayerPlan,
	operands layerOperands,
	execution *layerExecution,
) (DenseBlockResult, error) {
	c := options.Context
	switch plan.Mixer {
	case recurrentMixerSelectiveScan, recurrentMixerWeightedSelectiveScan:
		return buildSelectiveScanMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan.Mixer,
		)
	case recurrentMixerGroupedSelectiveScan, recurrentMixerScaledGroupedSelectiveScan,
		recurrentMixerSparseGroupedSelectiveScan:
		if plan.Mixer == recurrentMixerGroupedSelectiveScan && options.Weights.SSMConv1DBias == nil {
			return DenseBlockResult{}, errors.New("grouped selective-scan convolution bias is nil")
		}
		return buildGroupedSelectiveScanMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan.Mixer,
		)
	case recurrentMixerNormalizedSelectiveScan:
		return buildNormalizedSelectiveScanMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue],
		)
	case recurrentMixerGatedDelta:
		return buildGatedDeltaMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			c.Positions, c.Sequences,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue],
			plan.AttentionGraph.deltaProjection,
		)
	case recurrentMixerShortConvolution:
		return buildShortConvolutionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue],
		)
	case recurrentMixerDynamicWKV6:
		return buildDynamicWKV6MixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue],
		)
	case recurrentMixerAffineWKV6:
		return buildAffineWKV6MixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan,
		)
	case recurrentMixerDynamicWKV7:
		return buildDynamicWKV7MixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], plan,
		)
	case recurrentMixerKeyedDelta:
		return buildKeyedDeltaAttentionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue],
		)
	default:
		return DenseBlockResult{}, errors.New("compiled recurrent-mixing policy is invalid")
	}
}

func executeLayerProgram(
	options blockDispatchOptions,
	plan LayerPlan,
) (DenseBlockResult, error) {
	execution := layerExecution{
		residual: options.Context.Input, current: options.Context.Input,
	}
	for _, instruction := range plan.Program.Instructions {
		operands := resolveLayerOperands(options.Context, instruction)
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
	options blockDispatchOptions,
	plan LayerPlan,
	instruction LayerOperatorInstruction,
	operands layerOperands,
	execution *layerExecution,
) error {
	c := options.Context
	switch instruction.Operator {
	case LayerOperatorAttentionInputNorm:
		attentionInput, feedForwardBase, err := preparePolicyAttentionInputs(options)
		if err != nil {
			return err
		}
		execution.current = attentionInput
		execution.attentionInput = attentionInput
		execution.feedForwardBase = feedForwardBase
		return nil
	case LayerOperatorAttentionNorm:
		if options.Weights.AttentionNorm == nil {
			return errors.New("compiled attention-normalization stage is invalid")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.current, options.Weights.AttentionNorm,
			options.Weights.AttentionNormBias,
		)
		return c.Builder.Err()
	case LayerOperatorCrossAttentionNorm:
		if options.Weights.CrossAttentionNorm == nil {
			return errors.New("compiled cross-attention normalization stage is invalid")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.current, options.Weights.CrossAttentionNorm, nil,
		)
		return c.Builder.Err()
	case LayerOperatorRMSNorm:
		if execution.current == nil {
			return errors.New("compiled RMS-normalization stage is invalid")
		}
		execution.current = c.Builder.RMSNorm(execution.current, options.Spec.RMSNormEpsilon)
		return c.Builder.Err()
	case LayerOperatorPairedInputNorm:
		target := operands.tensors[RuntimeTensorPerLayerInput]
		validInput := false
		if execution.current != nil {
			_, validInput = tensor.MatrixRows(execution.current.Shape, uint64(options.Spec.EmbeddingLength))
		}
		if target == nil || execution.current == nil || !execution.current.Shape.Equal(target.Shape) ||
			!validInput ||
			options.Weights.AttentionNorm == nil || options.Weights.AttentionNorm2 == nil {
			return errors.New("compiled paired-input normalization stage is invalid")
		}
		tokenNorm := c.Builder.WeightedRMSNorm(
			execution.current, options.Weights.AttentionNorm, options.Spec.RMSNormEpsilon,
		)
		targetNorm := c.Builder.WeightedRMSNorm(
			target, options.Weights.AttentionNorm2, options.Spec.RMSNormEpsilon,
		)
		execution.current = c.Builder.Concat(tokenNorm, targetNorm, tensor.FirstOffset)
		execution.residual = target
		if options.Spec.NormBeforeResidual {
			execution.residual = targetNorm
		}
		return c.Builder.Err()
	case LayerOperatorAttentionPostNorm:
		if options.Weights.AttentionPostNorm == nil {
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
		result, err := executeAttentionOperator(options, plan, instruction.Operator, operands, execution)
		if err != nil {
			return err
		}
		mergeLayerResult(&execution.result, result)
		execution.current = result.Output
		return nil
	case LayerOperatorHybridMix:
		result, err := buildAttentionSSMHybridMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue], operands.caches[RuntimeCacheConvolution], operands.caches[RuntimeCacheSSM], plan,
		)
		if err != nil {
			return err
		}
		mergeLayerResult(&execution.result, result)
		execution.current = result.Output
		return nil
	case LayerOperatorRecurrentMix:
		result, err := executeRecurrentOperator(options, plan, operands, execution)
		if err != nil {
			return err
		}
		mergeLayerResult(&execution.result, result)
		execution.current = result.Output
		return nil
	case LayerOperatorFeedForwardNorm:
		if options.Weights.FeedForwardNorm == nil {
			return errors.New("compiled feed-forward normalization stage is invalid")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.residual, options.Weights.FeedForwardNorm,
			options.Weights.FeedForwardNormBias,
		)
		return c.Builder.Err()
	case LayerOperatorFeedForwardInputNorm:
		if execution.attentionInput == nil || execution.feedForwardBase == nil {
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
	case LayerOperatorFeedForwardPolicy, LayerOperatorFeedForwardRoutedSquaredReLU,
		LayerOperatorFeedForwardRoutedSwiGLU,
		LayerOperatorFeedForwardParallelGatedGELU, LayerOperatorFeedForwardEncoder,
		LayerOperatorFeedForwardRelative:
		var feedForward *tensor.Tensor
		var err error
		switch instruction.Operator {
		case LayerOperatorFeedForwardPolicy:
			feedForward, err = buildPolicyFeedForwardMix(
				options, execution.current, execution.residual,
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
		case LayerOperatorFeedForwardParallelGatedGELU:
			feedForward, err = buildParallelGatedGELUFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights, plan,
			)
		case LayerOperatorFeedForwardEncoder:
			feedForward, err = buildEncoderFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights, plan,
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
		if execution.current == nil {
			return errors.New("compiled feed-forward output stage is invalid")
		}
		if options.Weights.FeedForwardRouter == nil {
			execution.current = plan.ResidualStages.ApplyFeedForwardOutput(
				c.Builder, execution.current, options.Spec, options.Weights, plan.Normalization,
			)
		}
		return c.Builder.Err()
	case LayerOperatorFeedForwardPostNorm:
		if options.Weights.FeedForwardPostNorm == nil {
			return errors.New("compiled feed-forward post-normalization stage is invalid")
		}
		execution.current = c.Builder.WeightedRMSNorm(
			execution.current, options.Weights.FeedForwardPostNorm, options.Spec.RMSNormEpsilon,
		)
		return c.Builder.Err()
	case LayerOperatorCacheSentinel:
		tokens, validInput := tensor.MatrixRows(execution.residual.Shape, uint64(options.Spec.EmbeddingLength))
		if !validInput || uint64(len(c.Positions)) != tokens {
			return errors.New("compiled sentinel-cache positions are invalid")
		}
		key, value, err := buildSentinelCache(
			c.Builder, execution.residual, operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue],
		)
		if err != nil {
			return err
		}
		execution.result.Key = key
		execution.result.Value = value
		execution.result.Output = execution.residual
		return nil
	case LayerOperatorScale:
		if !positiveFinite(plan.ResidualStages.residualScale) {
			return errors.New("compiled scale stage is invalid")
		}
		execution.current = c.Builder.Scale(execution.current, plan.ResidualStages.residualScale)
		execution.result.Output = execution.current
		return c.Builder.Err()
	case LayerOperatorResidual:
		if execution.current == nil || execution.residual == nil {
			return errors.New("compiled residual stage is invalid")
		}
		execution.current = c.Builder.Add(execution.residual, execution.current)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case LayerOperatorResidualScale:
		if execution.current == nil || execution.residual == nil || options.Weights.LayerOutputScale == nil {
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
		if execution.current == nil || execution.residual == nil {
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
		if execution.current == nil || options.Weights.EmbeddingSkip == nil || options.Weights.LayerOutputScale == nil ||
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
		if execution.current == nil {
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
		if execution.current == nil {
			return errors.New("compiled token-shift stage is invalid")
		}
		result, err := buildTokenShiftFeedForwardMix(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[RuntimeCachePrimaryKey], execution.result.Key, instruction.Operator, plan.Normalization,
			plan.RecurrentRuntime.TokenShiftCount,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result.Key = result.Key
		return nil
	case LayerOperatorPeriodicScale:
		if execution.current == nil || !positiveFinite(plan.PeriodicScale) {
			return errors.New("compiled periodic-scale stage is invalid")
		}
		execution.current = c.Builder.Scale(execution.current, plan.PeriodicScale)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case LayerOperatorLatentAttention:
		result, err := buildLatentAttentionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], operands.caches[RuntimeCachePrimaryValue],
			operands.caches[RuntimeCacheIndexerKey], operands.tensors[RuntimeTensorPerLayerInput], plan,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	case LayerOperatorHyperAttention:
		result, err := buildHyperAttentionStage(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions,
			operands.caches[RuntimeCachePrimaryKey], c.PastStates,
			operands.tensors[RuntimeTensorCurrentPositions], plan,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	case LayerOperatorHyperFeedForward:
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
	if input == nil {
		return nil, nil, errors.New("sentinel-cache input is invalid")
	}
	width, _, validInput := tensor.MatrixExtents(input.Shape)
	if !validInput {
		return nil, nil, errors.New("sentinel-cache input is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "sentinel cache must contain both tensors"); err != nil {
		return nil, nil, err
	}
	// Distinct nodes: retained outputs require unique cache tensors.
	sentinel := func() *tensor.Tensor {
		return builder.GroupSlice(
			input, tensor.FirstOffset, tensor.SingletonExtent, tensor.SingletonExtent, width,
		)
	}
	cacheKey, cacheValue := sentinel(), sentinel()
	if pastKey != nil {
		_, cacheAxis, validCache := tensor.TrailingExtent(
			pastKey.Shape, tensor.SingletonExtent, tensor.SingletonExtent,
		)
		if !validCache || !pastKey.Shape.Equal(pastValue.Shape) {
			return nil, nil, errors.New("sentinel cache shape is invalid")
		}
		cacheKey = builder.Concat(pastKey, cacheKey, cacheAxis)
		cacheValue = builder.Concat(pastValue, cacheValue, cacheAxis)
	}
	return cacheKey, cacheValue, builder.Err()
}
