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

// BlockDispatchOptions: compiled layer-program inputs.
type BlockDispatchOptions struct {
	Context CachedBlockContext
	Spec    Spec
	Weights LayerGraphWeights
	Plan    *LayerPlan
}

// BuildArchitectureBlockCached: compiled layer-program construction.
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
	if context.MultiPositions != nil {
		if !options.Spec.SupportsMultiAxisPositions() {
			return DenseBlockResult{}, errors.New("layer architecture does not support multi-axis positions")
		}
		context.Positions = context.MultiPositions[0]
		options.Context = context
	}
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
	dense    denseFeedForwardState
	hasDense bool
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
		execution.current = ApplyNormalization(
			c.Builder, execution.current, options.Weights.AttentionNorm,
			options.Weights.AttentionNormBias, options.Spec,
		)
		return c.Builder.Err()
	case LayerOperatorRMSNorm:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 || execution.current == nil {
			return errors.New("compiled RMS-normalization stage is invalid")
		}
		execution.current = c.Builder.RMSNorm(execution.current, options.Spec.RMSNormEpsilon)
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
	case LayerOperatorAttentionMix:
		cacheCount := uint8(2)
		if instruction.Attention == AttentionMixOutputProjection {
			cacheCount = 0
		}
		if instruction.TensorCount != 0 || instruction.CacheCount != cacheCount {
			return errors.New("compiled attention-mixing stage is invalid")
		}
		var result DenseBlockResult
		var err error
		switch instruction.Attention {
		case AttentionMixCausalProjection:
			result, err = buildCausalProjectionMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan.Layer,
			)
		case AttentionMixGatedProjection:
			sequences := c.Sequences
			if sequences == 0 {
				sequences = 1
			}
			result, err = buildGatedProjectionMixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				c.Positions, c.MultiPositions, sequences,
				operands.caches[0], operands.caches[1], c.CacheWrite,
			)
		case AttentionMixOutputProjection:
			if options.Weights.AttentionOutput == nil {
				return errors.New("compiled output-projection stage is invalid")
			}
			projected := c.Builder.MulMat(options.Weights.AttentionOutput, execution.current)
			if options.Weights.AttentionOutputBias != nil {
				projected = c.Builder.Add(projected, options.Weights.AttentionOutputBias)
			}
			result.Output, err = projected, c.Builder.Err()
		case AttentionMixBidirectionalFusedQKV:
			result, err = buildBidirectionalFusedQKVMix(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan.Layer,
			)
		case AttentionMixBidirectionalQKNorm:
			result, err = buildBidirectionalQKNormMix(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan.Layer,
			)
		case AttentionMixCausalPostQKNorm:
			result, err = buildCausalPostQKNormMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], plan.Layer,
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
		execution.current = result.Output
		return nil
	case LayerOperatorHybridMix:
		if instruction.CacheCount != 4 || instruction.TensorCount != 0 {
			return errors.New("compiled hybrid-mixing stage is invalid")
		}
		var result DenseBlockResult
		var err error
		switch instruction.Hybrid {
		case HybridMixAttentionSSM:
			result, err = buildAttentionSSMHybridMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1], operands.caches[2], operands.caches[3],
			)
		default:
			return errors.New("compiled hybrid-mixing policy is invalid")
		}
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
		switch instruction.Recurrent {
		case RecurrentMixMamba:
			result, err = buildMambaMixerCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1],
			)
		case RecurrentMixMamba2:
			if instruction.RequireConvolutionBias && options.Weights.SSMConv1DBias == nil {
				return errors.New("Mamba2 recurrent-mixing convolution bias is nil")
			}
			result, err = buildMamba2MixerCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1],
			)
		case RecurrentMixPLaMo2:
			result, err = buildPLaMo2MixerCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				operands.caches[0], operands.caches[1],
			)
		case RecurrentMixGatedDelta:
			sequences := c.Sequences
			if sequences == 0 {
				sequences = 1
			}
			result, err = buildGatedDeltaMixCached(
				c.Builder, execution.current, options.Spec, options.Weights,
				c.Positions, sequences, operands.caches[0], operands.caches[1],
			)
		case RecurrentMixShortConvolution:
			result, err = buildShortConvolutionMixCached(
				c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
				operands.caches[0], operands.caches[1],
			)
		case RecurrentMixDynamicWKV6:
			result, err = buildDynamicWKV6MixCached(
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
		execution.current = ApplyNormalization(
			c.Builder, execution.residual, options.Weights.FeedForwardNorm,
			options.Weights.FeedForwardNormBias, options.Spec,
		)
		return c.Builder.Err()
	case LayerOperatorFeedForwardMix:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 {
			return errors.New("compiled feed-forward stage is invalid")
		}
		var feedForward *tensor.Tensor
		var err error
		switch instruction.FeedForward {
		case FeedForwardMixStandardSwiGLU:
			feedForward, err = buildStandardFeedForwardMix(
				c.Builder, execution.current, plan, options.Spec, options.Weights,
			)
		case FeedForwardMixFusedGLU:
			feedForward, err = buildFusedFeedForwardMix(
				c.Builder, execution.current, options.Spec, options.Weights, plan.Layer,
			)
		case FeedForwardMixSquaredReLU:
			feedForward, err = buildSquaredReLUFeedForwardMix(
				c.Builder, execution.current, options.Weights,
			)
		case FeedForwardMixRoutedSquaredReLU:
			feedForward, err = buildRoutedSquaredReLUFeedForwardMix(
				c.Builder, execution.current, options.Weights, plan.Experts,
			)
		case FeedForwardMixRoutedSwiGLU:
			feedForward, err = buildRoutedSwiGLUFeedForwardMix(
				c.Builder, execution.current, options.Weights,
				plan.Experts, plan.ExpertComposition,
			)
		case FeedForwardMixGatedGELU:
			feedForward, err = buildGatedGELUFeedForwardMix(
				c.Builder, execution.current, options.Weights,
			)
		default:
			return errors.New("compiled feed-forward policy is invalid")
		}
		if err != nil {
			return err
		}
		execution.current = feedForward
		return nil
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
	case LayerOperatorDenseTransformer:
		if instruction.CacheCount != 2 || instruction.TensorCount != 0 || plan.Block != BlockDense {
			return errors.New("compiled dense-transformer stage is invalid")
		}
		result, err := buildDenseBlock(options)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	case LayerOperatorDenseAttention:
		if instruction.CacheCount != 2 || instruction.TensorCount != 0 ||
			plan.Block != BlockDense {
			return errors.New("compiled dense-attention stage is invalid")
		}
		result, state, err := buildDenseAttentionStage(options)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		execution.dense = state
		execution.hasDense = true
		return nil
	case LayerOperatorDenseFeedForward:
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			plan.Block != BlockDense || !execution.hasDense {
			return errors.New("compiled dense feed-forward stage is invalid")
		}
		result, err := buildDenseFeedForwardStage(execution.dense)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		execution.hasDense = false
		return nil
	case LayerOperatorLinearAttention:
		if instruction.CacheCount != 2 || instruction.TensorCount != 0 ||
			plan.Block != BlockKimiLinear || !plan.Recurrent {
			return errors.New("compiled linear-attention stage is invalid")
		}
		result, err := buildKimiKDAMixCached(
			c.Builder, execution.current, options.Spec, options.Weights, c.Positions,
			operands.caches[0], operands.caches[1],
		)
		if err != nil {
			return err
		}
		execution.current = result.Output
		execution.result = result
		return nil
	case LayerOperatorLatentAttention:
		valid := instruction.CacheCount == 2 && instruction.TensorCount == 0 &&
			(plan.Block == BlockMLA || plan.Block == BlockKimiLinear && !plan.Recurrent)
		if plan.Block == BlockDSA {
			valid = instruction.CacheCount == 3 && instruction.TensorCount == 1
		}
		if !valid {
			return errors.New("compiled latent-attention stage is invalid")
		}
		result, err := buildLatentAttentionMixCachedWithPlan(
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
		if instruction.CacheCount != 1 || instruction.TensorCount != 1 || plan.Block != BlockDeepSeek4 {
			return errors.New("compiled hyper-attention stage is invalid")
		}
		result, err := buildDeepSeek4AttentionCachedWithPlan(
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
		if instruction.CacheCount != 0 || instruction.TensorCount != 0 ||
			plan.Block != BlockDeepSeek4 {
			return errors.New("compiled hyper-feed-forward stage is invalid")
		}
		output, err := buildDeepSeek4FeedForwardWithPlan(
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
		} else if plan.Block != BlockGraniteHybrid {
			required.add("feed-forward expert gate", weights.FeedForwardGateExperts)
			required.add("feed-forward expert up", weights.FeedForwardUpExperts)
		} else {
			required.add("feed-forward expert up", weights.FeedForwardUpExperts)
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
