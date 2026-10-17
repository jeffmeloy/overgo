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
	instruction layerOperatorInstruction,
) layerOperands {
	var operands layerOperands
	for index := range instruction.Caches[:instruction.CacheCount] {
		binding := instruction.Caches[index]
		switch binding {
		case runtimeCachePrimaryKey:
			operands.caches[binding] = context.PastKey
		case runtimeCachePrimaryValue:
			operands.caches[binding] = context.PastValue
		case runtimeCacheConvolution:
			operands.caches[binding] = context.PastStates[CacheStateConvolution].Value
		case runtimeCacheSSM:
			operands.caches[binding] = context.PastStates[CacheStateSSM].Value
		case runtimeCacheIndexerKey:
			operands.caches[binding] = context.PastStates[CacheStateIndexerKey].Value
		case runtimeCacheCrossKey:
			operands.caches[binding] = context.CrossKey
		case runtimeCacheCrossValue:
			operands.caches[binding] = context.CrossValue
		}
	}
	for index := range instruction.Tensors[:instruction.TensorCount] {
		binding := instruction.Tensors[index]
		switch binding {
		case runtimeTensorPerLayerInput:
			operands.tensors[binding] = context.PerLayerInput
		case runtimeTensorCurrentPositions:
			operands.tensors[binding] = context.CurrentPositions
		case runtimeTensorEncoder:
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
	injection       *tensor.Tensor
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
	operator layerOperator,
	operands layerOperands,
	execution *layerExecution,
) (DenseBlockResult, error) {
	c := options.Context
	switch operator {
	case layerOperatorAttentionCausalProjection:
		return buildCausalProjectionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan.Layer,
		)
	case layerOperatorAttentionGatedProjection:
		return buildGatedProjectionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			c.Positions, c.MultiPositions, c.Sequences,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], c.CacheWrite,
			plan.Rotary,
		)
	case layerOperatorAttentionOutputProjection:
		if options.Weights.AttentionOutput == nil {
			return DenseBlockResult{}, errors.New("compiled output-projection stage is invalid")
		}
		projected := c.Builder.MulMat(options.Weights.AttentionOutput, execution.current)
		if options.Weights.AttentionOutputBias != nil {
			projected = c.Builder.Add(projected, options.Weights.AttentionOutputBias)
		}
		return DenseBlockResult{Output: projected}, c.Builder.Err()
	case layerOperatorAttentionBidirectionalFusedQKV:
		return buildBidirectionalFusedQKVMix(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan,
		)
	case layerOperatorAttentionBidirectionalQKNorm:
		return buildBidirectionalQKNormMix(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan,
		)
	case layerOperatorAttentionCausalPostQKNorm:
		return buildCausalPostQKNormMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan.Layer,
		)
	case layerOperatorAttentionSharedKVQKNorm:
		return buildSharedKVQKNormMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan, c.CacheWrite,
		)
	case layerOperatorAttentionBidirectionalEncoder:
		return buildBidirectionalEncoderAttentionMix(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan,
		)
	case layerOperatorAttentionPairedCausalProjection:
		return buildPairedCausalProjectionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], c.CacheWrite,
		)
	case layerOperatorAttentionSharedCacheQKNorm:
		return buildSharedCacheQKNormMix(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan,
		)
	case layerOperatorAttentionPlannedProjection:
		return buildPolicyAttentionMix(
			options, execution.current, execution.feedForwardBase,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue],
		)
	case layerOperatorAttentionRelativeBidirectional:
		return buildRelativeSelfAttentionMix(
			c.Builder, execution.current, options.Spec, options.Weights, nil, nil, false,
		)
	case layerOperatorAttentionRelativeCausal:
		return buildRelativeSelfAttentionMix(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], true,
		)
	case layerOperatorAttentionCross:
		return buildCrossAttentionMix(
			c.Builder, execution.current, operands.tensors[runtimeTensorEncoder], options.Spec, options.Weights,
			operands.caches[runtimeCacheCrossKey], operands.caches[runtimeCacheCrossValue],
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
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan.Mixer,
		)
	case recurrentMixerGroupedSelectiveScan, recurrentMixerScaledGroupedSelectiveScan,
		recurrentMixerSparseGroupedSelectiveScan:
		if plan.Mixer == recurrentMixerGroupedSelectiveScan && options.Weights.SSMConv1DBias == nil {
			return DenseBlockResult{}, errors.New("grouped selective-scan convolution bias is nil")
		}
		return buildGroupedSelectiveScanMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan.Mixer,
		)
	case recurrentMixerNormalizedSelectiveScan:
		return buildNormalizedSelectiveScanMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue],
		)
	case recurrentMixerGatedDelta:
		return buildGatedDeltaMixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			c.Positions, c.Sequences,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue],
			plan.AttentionGraph.deltaProjection,
		)
	case recurrentMixerShortConvolution:
		return buildShortConvolutionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue],
		)
	case recurrentMixerDynamicWKV6:
		return buildDynamicWKV6MixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue],
		)
	case recurrentMixerAffineWKV6:
		return buildAffineWKV6MixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan,
		)
	case recurrentMixerDynamicWKV7:
		return buildDynamicWKV7MixCached(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], plan,
		)
	case recurrentMixerKeyedDelta:
		return buildKeyedDeltaAttentionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue],
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
	instruction layerOperatorInstruction,
	operands layerOperands,
	execution *layerExecution,
) error {
	c := options.Context
	switch instruction.Operator {
	case layerOperatorGatedHyperAttention, layerOperatorGatedHyperFeedForward, layerOperatorGatedHyperHead:
		return executeGatedHyperMix(options, instruction.Operator, execution)
	case layerOperatorGatedHyperCombine:
		if execution.injection == nil || execution.current == nil || execution.residual == nil {
			return errors.New("gated hyperconnection combination operands are nil")
		}
		execution.current = c.Builder.Add(execution.residual, c.Builder.Multiply(
			c.Builder.HyperConnectionInit(execution.current, options.Spec.HyperConnectionCount), execution.injection,
		))
		execution.result.Output = execution.current
		return c.Builder.Err()
	case layerOperatorAttentionInputNorm:
		attentionInput, feedForwardBase, err := preparePolicyAttentionInputs(options)
		if err != nil {
			return err
		}
		execution.current = attentionInput
		execution.attentionInput = attentionInput
		execution.feedForwardBase = feedForwardBase
		return nil
	case layerOperatorAttentionNorm:
		if options.Weights.AttentionNorm == nil {
			return errors.New("compiled attention-normalization stage is invalid")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.current, options.Weights.AttentionNorm,
			options.Weights.AttentionNormBias,
		)
		return c.Builder.Err()
	case layerOperatorCrossAttentionNorm:
		if options.Weights.CrossAttentionNorm == nil {
			return errors.New("compiled cross-attention normalization stage is invalid")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.current, options.Weights.CrossAttentionNorm, nil,
		)
		return c.Builder.Err()
	case layerOperatorRMSNorm:
		if execution.current == nil {
			return errors.New("compiled RMS-normalization stage is invalid")
		}
		execution.current = c.Builder.RMSNorm(execution.current, options.Spec.RMSNormEpsilon)
		return c.Builder.Err()
	case layerOperatorPairedInputNorm:
		target := operands.tensors[runtimeTensorPerLayerInput]
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
	case layerOperatorAttentionPostNorm:
		if options.Weights.AttentionPostNorm == nil {
			return errors.New("compiled attention post-normalization stage is invalid")
		}
		execution.current = c.Builder.WeightedRMSNorm(
			execution.current, options.Weights.AttentionPostNorm, options.Spec.RMSNormEpsilon,
		)
		return c.Builder.Err()
	case layerOperatorAttentionCausalProjection, layerOperatorAttentionGatedProjection,
		layerOperatorAttentionOutputProjection, layerOperatorAttentionBidirectionalFusedQKV,
		layerOperatorAttentionBidirectionalQKNorm, layerOperatorAttentionCausalPostQKNorm,
		layerOperatorAttentionSharedKVQKNorm, layerOperatorAttentionBidirectionalEncoder,
		layerOperatorAttentionPairedCausalProjection, layerOperatorAttentionSharedCacheQKNorm,
		layerOperatorAttentionPlannedProjection, layerOperatorAttentionRelativeBidirectional,
		layerOperatorAttentionRelativeCausal, layerOperatorAttentionCross:
		result, err := executeAttentionOperator(options, plan, instruction.Operator, operands, execution)
		if err != nil {
			return err
		}
		mergeLayerResult(&execution.result, result)
		execution.current = result.Output
		return nil
	case layerOperatorHybridMix:
		result, err := buildAttentionSSMHybridMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue], operands.caches[runtimeCacheConvolution], operands.caches[runtimeCacheSSM], plan,
		)
		if err != nil {
			return err
		}
		mergeLayerResult(&execution.result, result)
		execution.current = result.Output
		return nil
	case layerOperatorRecurrentMix:
		result, err := executeRecurrentOperator(options, plan, operands, execution)
		if err != nil {
			return err
		}
		mergeLayerResult(&execution.result, result)
		execution.current = result.Output
		return nil
	case layerOperatorFeedForwardNorm:
		if options.Weights.FeedForwardNorm == nil {
			return errors.New("compiled feed-forward normalization stage is invalid")
		}
		execution.current = plan.Normalization.Apply(
			c.Builder, execution.residual, options.Weights.FeedForwardNorm,
			options.Weights.FeedForwardNormBias,
		)
		return c.Builder.Err()
	case layerOperatorFeedForwardInputNorm:
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
	case layerOperatorFeedForwardPolicy, layerOperatorFeedForwardRoutedSquaredReLU,
		layerOperatorFeedForwardRoutedSwiGLU,
		layerOperatorFeedForwardParallelGatedGELU, layerOperatorFeedForwardEncoder,
		layerOperatorFeedForwardRelative:
		var feedForward *tensor.Tensor
		var err error
		switch instruction.Operator {
		case layerOperatorFeedForwardPolicy:
			feedForward, err = buildPolicyFeedForwardMix(
				options, execution.current, execution.residual,
			)
		case layerOperatorFeedForwardRoutedSquaredReLU:
			feedForward, err = buildRoutedSquaredReLUFeedForwardMix(
				c.Builder, execution.current, options.Weights, plan.Experts,
			)
		case layerOperatorFeedForwardRoutedSwiGLU:
			feedForward, err = buildRoutedSwiGLUFeedForwardMix(
				c.Builder, execution.current, options.Weights,
				plan.Experts, plan.ExpertComposition,
			)
		case layerOperatorFeedForwardParallelGatedGELU:
			feedForward, err = buildParallelGatedGELUFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights, plan,
			)
		case layerOperatorFeedForwardEncoder:
			feedForward, err = buildEncoderFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights, plan,
			)
		case layerOperatorFeedForwardRelative:
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
	case layerOperatorFeedForwardOutput:
		if execution.current == nil {
			return errors.New("compiled feed-forward output stage is invalid")
		}
		if options.Weights.FeedForwardRouter == nil {
			execution.current = plan.ResidualStages.ApplyFeedForwardOutput(
				c.Builder, execution.current, options.Spec, options.Weights, plan.Normalization,
			)
		}
		return c.Builder.Err()
	case layerOperatorFeedForwardPostNorm:
		if options.Weights.FeedForwardPostNorm == nil {
			return errors.New("compiled feed-forward post-normalization stage is invalid")
		}
		execution.current = c.Builder.WeightedRMSNorm(
			execution.current, options.Weights.FeedForwardPostNorm, options.Spec.RMSNormEpsilon,
		)
		return c.Builder.Err()
	case layerOperatorCacheSentinel:
		tokens, validInput := tensor.MatrixRows(execution.residual.Shape, uint64(options.Spec.EmbeddingLength))
		if !validInput || uint64(len(c.Positions)) != tokens {
			return errors.New("compiled sentinel-cache positions are invalid")
		}
		key, value, err := buildSentinelCache(
			c.Builder, execution.residual, operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue],
		)
		if err != nil {
			return err
		}
		execution.result.Key = key
		execution.result.Value = value
		execution.result.Output = execution.residual
		return nil
	case layerOperatorScale:
		if !positiveFinite(instruction.Scalar) {
			return errors.New("compiled scale stage is invalid")
		}
		execution.current = c.Builder.Scale(execution.current, instruction.Scalar)
		execution.result.Output = execution.current
		return c.Builder.Err()
	case layerOperatorResidual:
		if execution.current == nil || execution.residual == nil {
			return errors.New("compiled residual stage is invalid")
		}
		execution.current = c.Builder.Add(execution.residual, execution.current)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case layerOperatorResidualScale:
		if execution.current == nil || execution.residual == nil || options.Weights.LayerOutputScale == nil {
			return errors.New("compiled residual-scale stage is invalid")
		}
		execution.current = c.Builder.Multiply(
			c.Builder.Add(execution.residual, execution.current), options.Weights.LayerOutputScale,
		)
		execution.residual = execution.current
		execution.result.Output = execution.current
		return c.Builder.Err()
	case layerOperatorAttentionResidualNorm, layerOperatorInputResidualNorm,
		layerOperatorFeedForwardResidualNorm:
		if execution.current == nil || execution.residual == nil {
			return errors.New("compiled residual-normalization stage is invalid")
		}
		residual, weight, bias := execution.residual, options.Weights.AttentionPostNorm,
			options.Weights.AttentionPostNormBias
		switch instruction.Operator {
		case layerOperatorInputResidualNorm:
			residual, weight, bias = c.Input, options.Weights.AttentionNorm2, options.Weights.AttentionNorm2Bias
		case layerOperatorFeedForwardResidualNorm:
			weight, bias = options.Weights.FeedForwardPostNorm, options.Weights.FeedForwardPostNormBias
		}
		if instruction.Operator == layerOperatorInputResidualNorm && weight == nil && bias == nil {
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
	case layerOperatorScaledSkip:
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
	case layerOperatorOutputAdapter:
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
	case layerOperatorGatedTokenShiftSquaredReLU, layerOperatorTokenShiftSquaredReLU:
		if execution.current == nil {
			return errors.New("compiled token-shift stage is invalid")
		}
		result, err := buildTokenShiftFeedForwardMix(
			c.Builder, execution.current, options.Spec, options.Weights,
			operands.caches[runtimeCachePrimaryKey], execution.result.Key, instruction.Operator, plan.Normalization,
			plan.RecurrentRuntime.TokenShiftCount,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result.Key = result.Key
		return nil
	case layerOperatorLatentAttention:
		result, err := buildLatentAttentionMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], operands.caches[runtimeCachePrimaryValue],
			operands.caches[runtimeCacheIndexerKey], operands.tensors[runtimeTensorPerLayerInput], plan,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	case layerOperatorHyperAttention:
		result, err := buildHyperAttentionStage(
			c.Builder, c.Input, options.Spec, options.Weights, c.Positions,
			operands.caches[runtimeCachePrimaryKey], c.PastStates,
			operands.tensors[runtimeTensorCurrentPositions], plan,
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	case layerOperatorHyperFeedForward:
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
