package model

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

// Apply: learned normalization stage.
func (p NormalizationPlan) Apply(
	builder *tensor.Builder,
	input, weight, bias *tensor.Tensor,
) *tensor.Tensor {
	if p.Operation == NormalizationUnweightedLayer {
		return builder.LayerNorm(input, p.Epsilon)
	}
	if p.Operation == NormalizationUnweightedRMS {
		return builder.RMSNorm(input, p.Epsilon)
	}
	if p.Operation == NormalizationWeightOnlyLayer {
		return builder.Multiply(builder.LayerNorm(input, p.Epsilon), weight)
	}
	if p.Operation == NormalizationLayer {
		if bias == nil {
			return builder.Multiply(builder.LayerNorm(input, p.Epsilon), weight)
		}
		return builder.AffineLayerNorm(input, weight, bias, p.Epsilon)
	}
	normalized := builder.WeightedRMSNorm(input, weight, p.Epsilon)
	if p.RMSBias && bias != nil {
		return builder.Add(normalized, bias)
	}
	return normalized
}

type DenseBlockResult struct {
	Output    *tensor.Tensor
	Key       *tensor.Tensor
	Value     *tensor.Tensor
	Auxiliary *tensor.Tensor
	States    CacheStates[*tensor.Tensor]
	// Optional exact-attention replay inputs.
	Query          *tensor.Tensor
	AttentionScale float32
}

type denseBlockContext struct {
	builder            *tensor.Builder
	input              *tensor.Tensor
	spec               Spec
	weights            LayerGraphWeights
	positions          []uint32
	multiPositions     *[tensor.MaxDimensions][]uint32
	pastKey, pastValue *tensor.Tensor
	plan               LayerPlan
	layer              uint32
	cacheWrite         tensor.CacheWriteMode
}

func newDenseBlockContext(options blockDispatchOptions) denseBlockContext {
	c := options.Context
	return denseBlockContext{
		builder: c.Builder, input: c.Input, spec: options.Spec, weights: options.Weights,
		positions: c.Positions, multiPositions: c.MultiPositions,
		pastKey: c.PastKey, pastValue: c.PastValue, layer: c.Layer, cacheWrite: c.CacheWrite,
		plan: *options.Plan,
	}
}

func preparePolicyAttentionInputs(
	options blockDispatchOptions,
) (*tensor.Tensor, *tensor.Tensor, error) {
	c := newDenseBlockContext(options)
	if c.builder == nil || c.input == nil {
		return nil, nil, errors.New("compiled attention-input stage is nil")
	}
	if err := c.builder.Err(); err != nil {
		return nil, nil, err
	}
	if c.plan.DeciSparse {
		return nil, nil, errors.New("compiled attention-input stage is incompatible")
	}
	if err := c.plan.ExpertComposition.Validate(c.spec); err != nil {
		return nil, nil, err
	}
	if c.plan.FeedForward == FeedForwardXIELU &&
		(int(c.layer) >= len(c.spec.XIELUAlphaN) || int(c.layer) >= len(c.spec.XIELUAlphaP) ||
			int(c.layer) >= len(c.spec.XIELUBeta) || int(c.layer) >= len(c.spec.XIELUEpsilon)) {
		return nil, nil, errors.New("Apertus xIELU parameters are missing for layer")
	}
	usesExperts := c.weights.FeedForwardRouter != nil
	if err := c.plan.DenseWeights.Validate(c.spec, c.weights, usesExperts, c.plan.FeedForward); err != nil {
		return nil, nil, err
	}
	inputTokens, validInput := tensor.MatrixRows(c.input.Shape, uint64(c.spec.EmbeddingLength))
	if !validInput || uint64(len(c.positions)) != inputTokens {
		return nil, nil, fmt.Errorf(
			"dense block has %d positions for %d tokens", len(c.positions), inputTokens,
		)
	}
	if err := requireTensorPair(c.pastKey, c.pastValue, "dense block past key/value cache must both be present"); err != nil {
		return nil, nil, err
	}
	if !c.plan.AttentionGraph.Causal && !c.plan.AllowNonCausalCache && c.pastKey != nil {
		return nil, nil, errors.New("non-causal dense block does not support a KV cache")
	}
	normalized := c.input
	if c.plan.Normalization.PreAttention && !c.spec.SandwichNorm {
		normalized = c.plan.Normalization.Apply(
			c.builder, c.input, c.weights.AttentionNorm, c.weights.AttentionNormBias,
		)
	}
	feedForwardNormalized := normalized
	if c.plan.DenseWeights.useSecondaryAttentionNorm && c.weights.AttentionNorm2 != nil {
		if c.weights.AttentionNorm2Bias == nil {
			normalized = c.builder.Multiply(
				c.builder.LayerNorm(c.input, c.spec.LayerNormEpsilon), c.weights.AttentionNorm2,
			)
		} else {
			normalized = c.plan.Normalization.Apply(
				c.builder, c.input, c.weights.AttentionNorm2, c.weights.AttentionNorm2Bias,
			)
		}
	}
	return normalized, feedForwardNormalized, c.builder.Err()
}

func buildPolicyAttentionMix(
	options blockDispatchOptions,
	normalized, gateInput, pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	c := newDenseBlockContext(options)
	c.pastKey, c.pastValue = pastKey, pastValue
	tokens := uint64(len(c.positions))
	runtime := denseBlockRuntime{
		builder: c.builder, spec: c.spec, weights: c.weights, plan: c.plan,
		layer: c.layer, tokens: tokens,
	}
	attentionGate := c.plan.AttentionOutput.PrepareGate(c.builder, gateInput, c.weights)
	query, key, value := runtime.projectAttention(normalized)
	query, key, err := c.plan.QKPreprocess.Apply(c.builder, query, key, c.spec, c.weights, c.plan.Normalization, qkProjection)
	if err != nil {
		return DenseBlockResult{}, err
	}
	headCount := c.spec.LayerHeadCount(c.layer)
	kvHeadCount := c.spec.LayerKVHeadCount(c.layer)
	query = c.builder.Reshape(query, uint64(c.spec.KeyLength), uint64(headCount), tokens)
	key = c.builder.Reshape(key, uint64(c.spec.KeyLength), uint64(kvHeadCount), tokens)
	value = c.builder.Reshape(value, uint64(c.spec.ValueLength), uint64(kvHeadCount), tokens)
	query, key, err = c.plan.QKPreprocess.Apply(c.builder, query, key, c.spec, c.weights, c.plan.Normalization, qkHeads)
	if err != nil {
		return DenseBlockResult{}, err
	}
	query, key = c.plan.Rotary.Apply(
		c.builder, query, key, c.positions, c.multiPositions, c.weights.RopeFactors,
	)
	query, key, err = c.plan.QKPreprocess.Apply(c.builder, query, key, c.spec, c.weights, c.plan.Normalization, qkPostRotary)
	if err != nil {
		return DenseBlockResult{}, err
	}
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if c.pastKey != nil {
		pastTokens, keyAxis, validKey := tensor.TrailingExtent32(
			c.pastKey.Shape, uint64(c.spec.KeyLength), uint64(kvHeadCount),
		)
		valueTokens, valueAxis, validValue := tensor.TrailingExtent32(
			c.pastValue.Shape, uint64(c.spec.ValueLength), uint64(kvHeadCount),
		)
		if !validKey || !validValue || pastTokens != valueTokens {
			return DenseBlockResult{}, errors.New("dense block KV cache shape is invalid")
		}
		queryStart = c.builder.CacheTokenOffset(pastTokens)
		cacheKey = c.builder.WriteCache(c.pastKey, key, keyAxis, c.cacheWrite)
		cacheValue = c.builder.WriteCache(c.pastValue, value, valueAxis, c.cacheWrite)
	}
	attentionScale := c.spec.resolvedAttentionScale(uint64(c.spec.KeyLength))
	query, attentionScale = c.plan.QueryScale.Apply(c.builder, query, c.spec, c.weights, attentionScale)
	attention := c.plan.AttentionGraph.Build(
		c.builder, query, cacheKey, cacheValue, c.weights.AttentionSinks, nil,
		attentionScale, queryStart,
	)
	attention = c.plan.AttentionOutput.ApplyGate(
		c.builder, attention, attentionGate, c.weights, headCount, tokens, attentionGateHeads,
	)
	attention = c.builder.Reshape(attention, uint64(headCount)*uint64(c.spec.ValueLength), tokens)
	attention = c.plan.AttentionOutput.ApplyGate(
		c.builder, attention, attentionGate, c.weights, headCount, tokens, attentionGateFlat,
	)
	attention, err = c.plan.AttentionOutput.ApplyProjection(c.builder, attention, c.spec, c.weights)
	if err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: attention, Key: cacheKey, Value: cacheValue,
		Query: query, AttentionScale: attentionScale,
	}, c.builder.Err()
}

func buildPolicyFeedForwardMix(
	options blockDispatchOptions,
	normalized, residual *tensor.Tensor,
) (*tensor.Tensor, error) {
	c := newDenseBlockContext(options)
	usesExperts := c.weights.FeedForwardRouter != nil
	if usesExperts && c.plan.ExpertComposition.kind == expertDenseRoutedSeparateNorm {
		feedForward, err := c.plan.ExpertComposition.Build(
			c.builder, c.input, residual, normalized, c.spec, c.weights, c.plan,
		)
		if err != nil {
			return nil, err
		}
		return feedForward, c.builder.Err()
	}
	var feedForward *tensor.Tensor
	var err error
	if usesExperts {
		feedForward, err = c.plan.ExpertComposition.Build(
			c.builder, c.input, residual, normalized, c.spec, c.weights, c.plan,
		)
	} else {
		runtime := denseBlockRuntime{
			builder: c.builder, spec: c.spec, weights: c.weights, plan: c.plan,
			layer: c.layer, tokens: c.input.Shape.RowCount(),
		}
		feedForward, err = runtime.buildFeedForward(normalized)
	}
	if err != nil {
		return nil, err
	}
	return feedForward, c.builder.Err()
}

type denseBlockRuntime struct {
	builder *tensor.Builder
	spec    Spec
	weights LayerGraphWeights
	plan    LayerPlan
	layer   uint32
	tokens  uint64
}

func (r denseBlockRuntime) projectAttention(normalized *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	if r.weights.AttentionQKV != nil {
		shapes := r.spec.TensorShapes(r.layer)
		queryLength := shapes.QueryProjectionWidth()
		keyLength := shapes.KeyProjectionWidth()
		valueLength := shapes.ValueProjectionWidth()
		mixed := r.builder.MulMat(r.weights.AttentionQKV, normalized)
		if r.weights.AttentionQKVBias != nil {
			mixed = r.builder.Add(mixed, r.weights.AttentionQKVBias)
		}
		mixed = optionalSymmetricClamp(r.builder, mixed, r.spec.AttentionClamp)
		stride := queryLength + keyLength + valueLength
		return r.builder.Reshape(
				r.builder.GroupSlice(mixed, tensor.FirstOffset, queryLength, tensor.SingletonExtent, stride), queryLength, r.tokens,
			), r.builder.Reshape(
				r.builder.GroupSlice(mixed, queryLength, keyLength, tensor.SingletonExtent, stride), keyLength, r.tokens,
			), r.builder.Reshape(
				r.builder.GroupSlice(mixed, queryLength+keyLength, valueLength, tensor.SingletonExtent, stride), valueLength, r.tokens,
			)
	}
	query := r.builder.MulMat(r.weights.AttentionQ, normalized)
	key := r.builder.MulMat(r.weights.AttentionK, normalized)
	value := r.builder.MulMat(r.weights.AttentionV, normalized)
	for _, item := range []struct {
		tensor **tensor.Tensor
		scale  *tensor.Tensor
		bias   *tensor.Tensor
	}{
		{&query, r.weights.AttentionQScale, r.weights.AttentionQBias},
		{&key, r.weights.AttentionKScale, r.weights.AttentionKBias},
		{&value, r.weights.AttentionVScale, r.weights.AttentionVBias},
	} {
		if item.scale != nil {
			*item.tensor = r.builder.Multiply(*item.tensor, item.scale)
		}
		if item.bias != nil {
			*item.tensor = r.builder.Add(*item.tensor, item.bias)
		}
		*item.tensor = optionalSymmetricClamp(r.builder, *item.tensor, r.spec.AttentionClamp)
	}
	return query, key, value
}

func optionalSymmetricClamp(builder *tensor.Builder, input *tensor.Tensor, limit float32) *tensor.Tensor {
	if !positiveFinite(limit) {
		return input
	}
	return builder.Clamp(input, -limit, limit)
}

func (r denseBlockRuntime) buildFeedForward(normalized *tensor.Tensor) (*tensor.Tensor, error) {
	up := r.builder.MulMat(r.weights.FeedForwardUp, normalized)
	if r.weights.FeedForwardUpScale != nil {
		up = r.builder.Multiply(up, r.weights.FeedForwardUpScale)
	}
	if r.weights.FeedForwardUpBias != nil {
		up = r.builder.Add(up, r.weights.FeedForwardUpBias)
	}
	var activation *tensor.Tensor
	switch r.plan.FeedForward {
	case FeedForwardFusedGateUp:
		width := uint64(r.spec.LayerFeedForwardLength(r.layer))
		stride := tensor.PairedExtent * width
		gate := r.builder.Reshape(r.builder.GroupSlice(
			up, tensor.FirstOffset, width, tensor.SingletonExtent, stride,
		), width, r.tokens)
		up = r.builder.Reshape(r.builder.GroupSlice(
			up, width, width, tensor.SingletonExtent, stride,
		), width, r.tokens)
		switch r.spec.HiddenActivation {
		case "reglu":
			activation = r.builder.ReGLU(gate, up)
		case "gelu", "geglu":
			activation = r.builder.GEGLU(gate, up)
		default:
			activation = r.builder.SwiGLU(gate, up)
		}
	case FeedForwardXIELU:
		activation = r.builder.XIELU(
			up, r.spec.XIELUAlphaN[r.layer], r.spec.XIELUAlphaP[r.layer],
			r.spec.XIELUBeta[r.layer], r.spec.XIELUEpsilon[r.layer],
		)
	case FeedForwardGELU, FeedForwardSequentialGELU:
		activation = r.builder.GELU(up)
	case FeedForwardSquaredReLU:
		activation = r.builder.ReLUSquared(up)
	case FeedForwardSwiGLU, FeedForwardGEGLU:
		gate := r.builder.MulMat(r.weights.FeedForwardGate, normalized)
		if r.weights.FeedForwardGateScale != nil {
			gate = r.builder.Multiply(gate, r.weights.FeedForwardGateScale)
		}
		if r.weights.FeedForwardGateBias != nil {
			gate = r.builder.Add(gate, r.weights.FeedForwardGateBias)
		}
		activation = r.builder.SwiGLU(gate, up)
		if r.plan.FeedForward == FeedForwardGEGLU {
			activation = r.builder.GEGLU(gate, up)
		}
	default:
		return nil, fmt.Errorf("unsupported feed-forward policy %d", r.plan.FeedForward)
	}
	if r.weights.FeedForwardActivationScale != nil {
		if !r.plan.DenseWeights.allowActivationScale {
			return nil, errors.New("feed-forward activation scale is not enabled")
		}
		activation = r.builder.Divide(activation, r.weights.FeedForwardActivationScale)
	}
	if r.plan.DenseWeights.requireSubNorm {
		activation = r.builder.WeightedRMSNorm(
			activation, r.weights.FeedForwardSubNorm, r.spec.RMSNormEpsilon,
		)
	}
	feedForward := r.builder.MulMat(r.weights.FeedForwardDown, activation)
	if r.weights.FeedForwardDownScale != nil {
		feedForward = r.builder.Multiply(feedForward, r.weights.FeedForwardDownScale)
	}
	if r.weights.FeedForwardDownBias != nil {
		feedForward = r.builder.Add(feedForward, r.weights.FeedForwardDownBias)
	}
	return feedForward, r.builder.Err()
}

func buildSharedKVQKNormMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerPlan LayerPlan,
	cacheWrite tensor.CacheWriteMode,
) (DenseBlockResult, error) {
	layerIndex := layerPlan.Layer
	tokenCount, validInput := tensor.MatrixRows32(normalized.Shape, uint64(spec.EmbeddingLength))
	if !validInput || uint64(len(positions)) != uint64(tokenCount) {
		return DenseBlockResult{}, errors.New("shared-KV attention input shape is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "shared-KV attention cache pair is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	required := graphWeights{
		weights.AttentionQ,
		weights.AttentionQNorm,
		weights.AttentionOutput,
	}
	if layerPlan.HasKV {
		required = append(required, weights.AttentionK)
		required = append(required, weights.AttentionKNorm)
	} else if pastKey == nil {
		return DenseBlockResult{}, errors.New("shared-KV attention has no source cache")
	}
	if err := required.validate("shared-KV attention"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(tokenCount)
	shapes := spec.TensorShapes(layerIndex)
	headCount, kvHeadCount := shapes.QueryHeads, shapes.KVHeads
	keyLength, valueLength := shapes.Key, shapes.Value
	query := builder.Reshape(
		builder.MulMat(weights.AttentionQ, normalized), keyLength, headCount, tokens,
	)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	query = layerPlan.Rotary.ApplyOne(builder, query, positions, nil, weights.RopeFactors)
	cacheKey, cacheValue := pastKey, pastValue
	queryStart := uint32(0)
	if layerPlan.HasKV {
		key := builder.Reshape(
			builder.MulMat(weights.AttentionK, normalized), keyLength, kvHeadCount, tokens,
		)
		value := key
		if weights.AttentionV != nil {
			value = builder.Reshape(
				builder.MulMat(weights.AttentionV, normalized), valueLength, kvHeadCount, tokens,
			)
		}
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
		value = builder.RMSNorm(value, spec.RMSNormEpsilon)
		key = layerPlan.Rotary.ApplyOne(builder, key, positions, nil, weights.RopeFactors)
		cacheKey, cacheValue = key, value
		if pastKey != nil {
			pastTokens, keyAxis, validKey := tensor.TrailingExtent32(pastKey.Shape, keyLength, kvHeadCount)
			valueTokens, valueAxis, validValue := tensor.TrailingExtent32(pastValue.Shape, valueLength, kvHeadCount)
			if !validKey || !validValue || pastTokens != valueTokens {
				return DenseBlockResult{}, errors.New("shared-KV attention cache shape is invalid")
			}
			queryStart = builder.CacheTokenOffset(pastTokens)
			cacheKey = builder.WriteCache(pastKey, key, keyAxis, cacheWrite)
			cacheValue = builder.WriteCache(pastValue, value, valueAxis, cacheWrite)
		}
	} else {
		pastTokens, _, validKey := tensor.TrailingExtent32(pastKey.Shape, keyLength, kvHeadCount)
		valueTokens, _, validValue := tensor.TrailingExtent32(pastValue.Shape, valueLength, kvHeadCount)
		if !validKey || !validValue || pastTokens != valueTokens || pastTokens < tokenCount {
			return DenseBlockResult{}, errors.New("shared-KV attention source shape is invalid")
		}
		queryStart = pastTokens - tokenCount
	}
	attention := layerPlan.AttentionGraph.Build(
		builder, query, cacheKey, cacheValue, nil, weights.AttentionBlockIDs,
		spec.AttentionScale, queryStart,
	)
	attention = builder.Reshape(attention, headCount*valueLength, tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: cacheKey, Value: cacheValue}, nil
}

func buildParallelGatedGELUFeedForwardMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	layerPlan LayerPlan,
) (*tensor.Tensor, error) {
	required := graphWeights{
		weights.FeedForwardNorm,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
		weights.FeedForwardPostNorm,
	}
	usesExperts := weights.FeedForwardRouter != nil
	if usesExperts {
		required = append(required, weights.FeedForwardRouter)
		required = append(required, weights.FeedForwardRouterScale)
		required = append(required, weights.FeedForwardDownExperts)
		required = append(required, weights.FeedForwardPreNorm2)
		required = append(required, weights.FeedForwardPostNorm1)
		required = append(required, weights.FeedForwardPostNorm2)
		if weights.FeedForwardGateUpExperts == nil {
			required = append(required, weights.FeedForwardGateExperts)
			required = append(required, weights.FeedForwardUpExperts)
		}
	}
	if err := required.validate("parallel gated feed-forward"); err != nil {
		return nil, err
	}
	var feedForward *tensor.Tensor
	if usesExperts {
		denseInput := builder.WeightedRMSNorm(input, weights.FeedForwardNorm, spec.RMSNormEpsilon)
		denseGate := builder.MulMat(weights.FeedForwardGate, denseInput)
		denseUp := builder.MulMat(weights.FeedForwardUp, denseInput)
		dense := builder.MulMat(weights.FeedForwardDown, builder.GEGLU(denseGate, denseUp))
		dense = builder.WeightedRMSNorm(dense, weights.FeedForwardPostNorm1, spec.RMSNormEpsilon)
		expertInput := builder.WeightedRMSNorm(input, weights.FeedForwardPreNorm2, spec.RMSNormEpsilon)
		routerInput := builder.Scale(builder.RMSNorm(input, spec.RMSNormEpsilon), hostmath.InvSqrt32(uint64(spec.EmbeddingLength)))
		routerInput = builder.Multiply(routerInput, weights.FeedForwardRouterScale)
		feedForward = layerPlan.Experts.BuildLayer(builder, expertInput, routerInput, weights)
		feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm2, spec.RMSNormEpsilon)
		feedForward = builder.Add(dense, feedForward)
	} else {
		feedForwardInput := builder.WeightedRMSNorm(input, weights.FeedForwardNorm, spec.RMSNormEpsilon)
		gate := builder.MulMat(weights.FeedForwardGate, feedForwardInput)
		up := builder.MulMat(weights.FeedForwardUp, feedForwardInput)
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up))
	}
	feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	return feedForward, builder.Err()
}

func buildOutputAdapter(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	layerPlan LayerPlan,
) (*tensor.Tensor, error) {
	output := input
	if layerPlan.PerLayerInput {
		required := graphWeights{
			weights.PerLayerInput,
			weights.PerLayerInputGate,
			weights.PerLayerProjection,
			weights.PerLayerPostNorm,
		}
		if err := required.validate("output adapter"); err != nil {
			return nil, err
		}
		perLayer := builder.GELU(builder.MulMat(weights.PerLayerInputGate, output))
		perLayer = builder.Multiply(perLayer, weights.PerLayerInput)
		perLayer = builder.MulMat(weights.PerLayerProjection, perLayer)
		perLayer = builder.WeightedRMSNorm(perLayer, weights.PerLayerPostNorm, spec.RMSNormEpsilon)
		output = builder.Add(output, perLayer)
	}
	if weights.LayerOutputScale != nil {
		output = builder.Multiply(output, weights.LayerOutputScale)
	}
	return output, builder.Err()
}

// ActivationProjectionResult: projected activation seam before host policy.
type ActivationProjectionResult struct {
	Residual *tensor.Tensor
	Gate     *tensor.Tensor
	Up       *tensor.Tensor
	Key      *tensor.Tensor
	Value    *tensor.Tensor
}

// BuildActivationProjection executes the program-owned projection prefix.
func (p CompiledLayerProgram) BuildActivationProjection(
	context CachedBlockContext,
	weights LayerGraphWeights,
) (ActivationProjectionResult, error) {
	if !p.plan.splitProjection() || p.plan.Layer != context.Layer ||
		p.plan.Recurrent != context.Recurrent {
		return ActivationProjectionResult{}, errors.New("compiled activation-projection program is incompatible")
	}
	builder, input, spec := context.Builder, context.Input, p.spec
	positions, pastKey, pastValue := context.Positions, context.PastKey, context.PastValue
	layerIndex := p.plan.Layer
	if builder == nil || input == nil {
		return ActivationProjectionResult{}, errors.New("activation-projection input is invalid")
	}
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput || len(positions) == 0 || uint64(len(positions)) != tokens {
		return ActivationProjectionResult{}, errors.New("activation-projection input is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "split-projection cache pair is incomplete"); err != nil {
		return ActivationProjectionResult{}, err
	}
	required := graphWeights{
		weights.AttentionNorm,
		weights.AttentionQ,
		weights.AttentionQNorm,
		weights.AttentionOutput,
		weights.AttentionPostNorm,
		weights.FeedForwardNorm,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.LaurelLeft,
		weights.LaurelRight,
		weights.LaurelPostNorm,
	}
	if p.plan.HasKV {
		required = append(required, weights.AttentionK)
		required = append(required, weights.AttentionV)
		required = append(required, weights.AttentionKNorm)
	} else if pastKey == nil {
		return ActivationProjectionResult{}, errors.New("split-projection shared-KV layer has no source cache")
	}
	if err := required.validate("split projection"); err != nil {
		return ActivationProjectionResult{}, err
	}
	shapes := spec.TensorShapes(layerIndex)
	headCount, kvHeadCount := shapes.QueryHeads, shapes.KVHeads
	keyLength, valueLength := shapes.Key, shapes.Value
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	laurel := builder.MulMat(weights.LaurelLeft, normalized)
	laurel = builder.MulMat(weights.LaurelRight, laurel)
	laurel = builder.WeightedRMSNorm(laurel, weights.LaurelPostNorm, spec.RMSNormEpsilon)
	laurel = builder.Add(laurel, normalized)
	query := builder.Reshape(builder.MulMat(weights.AttentionQ, normalized), keyLength, headCount, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	frequencyBase := spec.RopeFrequencyBase
	if p.plan.Sliding {
		frequencyBase = spec.RopeFrequencySWA
	}
	query = builder.RoPEWithOptions(query, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.LayerRopeDimensionCount(layerIndex), FrequencyBase: frequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
	cacheKey, cacheValue := pastKey, pastValue
	var queryStart uint32
	if p.plan.HasKV {
		key := builder.Reshape(builder.MulMat(weights.AttentionK, normalized), keyLength, kvHeadCount, tokens)
		value := builder.Reshape(builder.MulMat(weights.AttentionV, normalized), valueLength, kvHeadCount, tokens)
		key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
		value = builder.RMSNorm(value, spec.RMSNormEpsilon)
		key = builder.RoPEWithOptions(key, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.LayerRopeDimensionCount(layerIndex), FrequencyBase: frequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
		cacheKey, cacheValue = key, value
		if pastKey != nil {
			pastTokens, cacheAxis, keyOK := tensor.TrailingExtent(pastKey.Shape, keyLength, kvHeadCount)
			valueTokens, _, valueOK := tensor.TrailingExtent(pastValue.Shape, valueLength, kvHeadCount)
			if !keyOK || !valueOK || pastTokens != valueTokens || pastTokens > math.MaxUint32 {
				return ActivationProjectionResult{}, errors.New("split-projection cache shape is invalid")
			}
			queryStart = uint32(pastTokens)
			cacheKey = builder.Concat(pastKey, key, cacheAxis)
			cacheValue = builder.Concat(pastValue, value, cacheAxis)
		}
	} else {
		pastTokens, _, keyOK := tensor.TrailingExtent(pastKey.Shape, keyLength, kvHeadCount)
		valueTokens, _, valueOK := tensor.TrailingExtent(pastValue.Shape, valueLength, kvHeadCount)
		if !keyOK || !valueOK || pastTokens != valueTokens || pastTokens < tokens || pastTokens > math.MaxUint32 {
			return ActivationProjectionResult{}, errors.New("split-projection shared-KV source shape is invalid")
		}
		queryStart = uint32(pastTokens - tokens)
	}
	var attention *tensor.Tensor
	attentionScale := spec.resolvedAttentionScale(keyLength)
	query = builder.Scale(query, attentionScale)
	if p.plan.Sliding {
		attention = builder.AttentionWithOptions(
			query, cacheKey, cacheValue, tensor.AttentionOptions{Scale: tensor.UnitScale, Causal: true, QueryStart: queryStart, Window: spec.SlidingWindow})

	} else {
		attention = builder.AttentionWithOptions(query, cacheKey, cacheValue, tensor.AttentionOptions{Scale: tensor.UnitScale, Causal: true, QueryStart: queryStart})
	}
	attention = builder.MulMat(weights.AttentionOutput, builder.Reshape(attention, headCount*valueLength, tokens))
	attention = builder.WeightedRMSNorm(attention, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	residual := builder.Scale(builder.Add(builder.Add(input, attention), laurel), spec.Profile().Runtime.ActivationResidualScale)
	feedForwardInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	result := ActivationProjectionResult{
		Residual: residual,
		Gate:     builder.MulMat(weights.FeedForwardGate, feedForwardInput),
		Up:       builder.MulMat(weights.FeedForwardUp, feedForwardInput),
		Key:      cacheKey,
		Value:    cacheValue,
	}
	if err := builder.Err(); err != nil {
		return ActivationProjectionResult{}, err
	}
	return result, nil
}

// BuildActivatedOutput executes the program-owned projection suffix.
func (p CompiledLayerProgram) BuildActivatedOutput(
	builder *tensor.Builder,
	residual, activated *tensor.Tensor,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if !p.plan.splitProjection() || builder == nil || residual == nil || activated == nil ||
		weights.FeedForwardDown == nil || weights.FeedForwardPostNorm == nil {
		return nil, errors.New("compiled activated-output stage is incomplete")
	}
	output := builder.MulMat(weights.FeedForwardDown, activated)
	output = builder.WeightedRMSNorm(output, weights.FeedForwardPostNorm, p.spec.RMSNormEpsilon)
	output = builder.Add(residual, output)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildPerLayerInputs executes compiled per-layer embedding projections.
func (p ModelPlan) BuildPerLayerInputs(
	builder *tensor.Builder,
	input, tokenEmbedding, modelProjection, projectionNorm *tensor.Tensor,
) ([]*tensor.Tensor, error) {
	spec := p.spec
	if builder == nil || input == nil || tokenEmbedding == nil || modelProjection == nil || projectionNorm == nil {
		return nil, errors.New("mapped per-layer input is incomplete")
	}
	tokens, inputOK := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !p.profile.Has(ArchitecturePerLayerEmbeddings) || !inputOK {
		return nil, errors.New("mapped per-layer input configuration is invalid")
	}
	width := uint64(spec.EmbeddingPerLayer)
	layers := uint64(spec.BlockCount)
	combinedWidth := width * layers
	if !tensor.IsMatrix(tokenEmbedding.Shape, combinedWidth, tokens) ||
		!tensor.IsMatrix(modelProjection.Shape, uint64(spec.EmbeddingLength), combinedWidth) ||
		!tensor.IsVector(projectionNorm.Shape, width) {
		return nil, errors.New("mapped per-layer input shape is invalid")
	}
	projected := builder.MulMat(modelProjection, input)
	projected = builder.Scale(projected, hostmath.InvSqrt32(uint64(spec.EmbeddingLength)))
	projected = builder.Reshape(projected, width, layers*tokens)
	projected = builder.WeightedRMSNorm(projected, projectionNorm, spec.RMSNormEpsilon)
	selected := builder.Scale(tokenEmbedding, hostmath.Sqrt32(width))
	selected = builder.Reshape(selected, width, layers*tokens)
	combined := builder.Scale(builder.Add(projected, selected), p.profile.Runtime.ActivationResidualScale)
	combined = builder.Reshape(combined, combinedWidth, tokens)
	result := make([]*tensor.Tensor, spec.BlockCount)
	for layer := uint64(tensor.FirstOffset); layer < layers; layer++ {
		result[layer] = builder.Reshape(
			builder.GroupSlice(combined, layer*width, width, tensor.SingletonExtent, combinedWidth),
			width, tokens,
		)
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

func postActivationLimitedSwiGLU(builder *tensor.Builder, gate, up *tensor.Tensor, limit float32) *tensor.Tensor {
	if !positiveFinite(limit) {
		return builder.SwiGLU(gate, up)
	}
	up = builder.Clamp(up, -limit, limit)
	gate = builder.Clamp(builder.SiLU(gate), -math.MaxFloat32, limit)
	return builder.Multiply(gate, up)
}

func buildSharedSwiGLU(
	builder *tensor.Builder,
	input *tensor.Tensor,
	weights LayerGraphWeights,
) *tensor.Tensor {
	gate := builder.MulMat(weights.FeedForwardSharedGate, input)
	up := builder.MulMat(weights.FeedForwardSharedUp, input)
	return builder.MulMat(weights.FeedForwardSharedDown, builder.SwiGLU(gate, up))
}

func inputLimitedSwiGLU(builder *tensor.Builder, gate, up *tensor.Tensor, limit float32) *tensor.Tensor {
	if !positiveFinite(limit) {
		return builder.SwiGLU(gate, up)
	}
	gate = builder.Clamp(gate, -math.MaxFloat32, limit)
	up = builder.Clamp(up, -limit, limit)
	return builder.SwiGLU(gate, up)
}

func buildGatedProjectionMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	multiPositions *[tensor.MaxDimensions][]uint32,
	sequences uint64,
	pastKey, pastValue *tensor.Tensor,
	cacheWrite tensor.CacheWriteMode,
	deltaProjection gatedDeltaPolicy,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil {
		return DenseBlockResult{}, errors.New("gated-delta attention mix input is nil")
	}
	required := graphWeights{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.AttentionQNorm,
		weights.AttentionKNorm,
	}
	if err := required.validate("gated-delta attention mix"); err != nil {
		return DenseBlockResult{}, err
	}
	if !tensor.IsBatchedMatrix(
		normalized.Shape, uint64(spec.EmbeddingLength), uint64(len(positions)), sequences,
	) {
		return DenseBlockResult{}, errors.New("gated-delta attention position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "gated-delta attention cache must contain both key and value"); err != nil {
		return DenseBlockResult{}, err
	}

	tokens := uint64(len(positions))
	headWidth := uint64(spec.KeyLength)
	heads := uint64(spec.HeadCount)
	queryAndGate := builder.MulMat(weights.AttentionQ, normalized)
	query := builder.GroupSlice(queryAndGate, tensor.FirstOffset, headWidth, heads, tensor.PairedExtent*headWidth)
	gate := builder.GroupSlice(queryAndGate, headWidth, headWidth, heads, tensor.PairedExtent*headWidth)
	if sequences > tensor.SingletonExtent {
		query = builder.Reshape(query, headWidth, heads, tokens, sequences)
		gate = builder.Reshape(gate, headWidth, heads, tokens, sequences)
	}
	keyProjection := builder.MulMat(weights.AttentionK, normalized)
	valueProjection := builder.MulMat(weights.AttentionV, normalized)
	var key, value *tensor.Tensor
	if sequences == tensor.SingletonExtent {
		key = builder.Reshape(keyProjection, headWidth, uint64(spec.HeadCountKV), tokens)
		value = builder.Reshape(valueProjection, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	} else {
		key = builder.Reshape(keyProjection, headWidth, uint64(spec.HeadCountKV), tokens, sequences)
		value = builder.Reshape(
			valueProjection, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens, sequences,
		)
	}
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	frequencyScale := spec.ropeFrequencyScale()
	if deltaProjection == gatedDeltaInterleavedProjections {
		query = builder.RoPEWithOptions(
			query, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: frequencyScale})

		key = builder.RoPEWithOptions(
			key, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: frequencyScale})

	} else {
		resolved := [4][]uint32{}
		if multiPositions == nil {
			for axis := range resolved {
				resolved[axis] = positions
			}
		} else {
			resolved = *multiPositions
		}
		query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
			MultiPositions: &resolved, Sections: spec.RopeSections,
			RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase,
			FrequencyScale: frequencyScale, InterleavedSections: spec.Profile().Rotary.MultiAxis == multiAxisRotaryInterleaved,
		})
	}

	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		pastTokens, keyAxis, validKey := tensor.BatchedTrailingExtent32(
			pastKey.Shape, sequences, uint64(spec.KeyLength), uint64(spec.HeadCountKV),
		)
		valueTokens, valueAxis, validValue := tensor.BatchedTrailingExtent32(
			pastValue.Shape, sequences, uint64(spec.ValueLength), uint64(spec.HeadCountKV),
		)
		if !validKey || !validValue || pastTokens != valueTokens {
			return DenseBlockResult{}, errors.New("gated-delta attention cache shape is invalid")
		}
		queryStart = builder.CacheTokenOffset(pastTokens)
		cacheKey = builder.WriteCache(pastKey, key, keyAxis, cacheWrite)
		cacheValue = builder.WriteCache(pastValue, value, valueAxis, cacheWrite)
	}
	attentionScale := spec.resolvedAttentionScale(uint64(spec.KeyLength))
	attention := builder.AttentionWithOptions(
		query,
		cacheKey,
		cacheValue, tensor.AttentionOptions{Scale: attentionScale, Causal: true, QueryStart: queryStart})

	attention = builder.Reshape(
		attention,
		uint64(spec.HeadCount)*uint64(spec.ValueLength),
		tokens*sequences,
	)
	gate = builder.Reshape(gate, uint64(spec.HeadCount)*headWidth, tokens*sequences)
	attention = builder.Multiply(attention, builder.Sigmoid(gate))
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: cacheKey, Value: cacheValue}, nil
}

func buildGatedDeltaMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	sequences uint64,
	convState, ssmState *tensor.Tensor,
	deltaPolicy gatedDeltaPolicy,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("gated-delta recurrent mix input/state is nil")
	}
	required := graphWeights{
		weights.AttentionQKV,
		weights.SSMConv1D,
		weights.SSMTimeStep,
		weights.SSMA,
		weights.SSMNorm,
		weights.SSMOutput,
	}
	if deltaPolicy == gatedDeltaInterleavedProjections {
		required = append(required, weights.SSMBetaAlpha)
		if weights.AttentionGate != nil {
			required = append(required, weights.AttentionGate)
		}
	} else {
		required = append(required, weights.AttentionGate)
		required = append(required, weights.SSMBeta)
		required = append(required, weights.SSMAlpha)
	}
	if err := required.validate("gated-delta recurrent mix"); err != nil {
		return DenseBlockResult{}, err
	}
	if !tensor.IsBatchedMatrix(
		normalized.Shape, uint64(spec.EmbeddingLength), uint64(len(positions)), sequences,
	) {
		return DenseBlockResult{}, errors.New("gated-delta recurrent position count is invalid")
	}
	tokens := uint64(len(positions))
	stateWidth := uint64(spec.SSMStateSize)
	keyHeads := uint64(spec.SSMGroupCount)
	valueHeads := uint64(spec.SSMTimeStepRank)
	keyDimension := stateWidth * keyHeads
	valueDimension := uint64(spec.SSMInnerSize)
	convChannels := tensor.PairedExtent*keyDimension + valueDimension
	convStateWidth := uint64(spec.SSMConvKernel - tensor.SingletonExtent)
	validConvState := tensor.HasDimensions(convState.Shape, convStateWidth, convChannels)
	if sequences > tensor.SingletonExtent {
		validConvState = tensor.HasDimensions(convState.Shape, convStateWidth, convChannels, sequences)
	}
	if !validConvState ||
		!tensor.HasDimensions(ssmState.Shape, stateWidth, stateWidth, valueHeads, sequences) {
		return DenseBlockResult{}, errors.New("gated-delta recurrent cache shape is invalid")
	}

	qkvProjection := builder.MulMat(weights.AttentionQKV, normalized)
	qkvMixed := qkvProjection
	var z *tensor.Tensor
	if deltaPolicy == gatedDeltaInterleavedProjections && weights.AttentionGate == nil {
		valueHeadsPerGroup := valueHeads / keyHeads
		valueWidthPerGroup := stateWidth * valueHeadsPerGroup
		groupStride := tensor.PairedExtent*stateWidth + tensor.PairedExtent*valueWidthPerGroup
		query := builder.GroupSlice(qkvProjection, tensor.FirstOffset, stateWidth, keyHeads, groupStride)
		key := builder.GroupSlice(qkvProjection, stateWidth, stateWidth, keyHeads, groupStride)
		value := builder.GroupSlice(
			qkvProjection, tensor.PairedExtent*stateWidth, valueWidthPerGroup, keyHeads, groupStride,
		)
		z = builder.GroupSlice(
			qkvProjection, tensor.PairedExtent*stateWidth+valueWidthPerGroup,
			valueWidthPerGroup, keyHeads, groupStride,
		)
		qkvMixed = builder.Concat(
			builder.Concat(
				builder.Reshape(query, keyDimension, tokens),
				builder.Reshape(key, keyDimension, tokens),
				tensor.FirstOffset,
			),
			builder.Reshape(value, valueDimension, tokens),
			tensor.FirstOffset,
		)
		z = builder.Reshape(z, stateWidth, valueHeads, tokens, sequences)
	} else {
		z = builder.MulMat(weights.AttentionGate, normalized)
	}
	var beta, alpha *tensor.Tensor
	if deltaPolicy == gatedDeltaInterleavedProjections {
		valueHeadsPerGroup := valueHeads / keyHeads
		betaAlpha := builder.MulMat(weights.SSMBetaAlpha, normalized)
		beta = builder.GroupSlice(
			betaAlpha, tensor.FirstOffset, valueHeadsPerGroup, keyHeads, tensor.PairedExtent*valueHeadsPerGroup,
		)
		alpha = builder.GroupSlice(
			betaAlpha, valueHeadsPerGroup, valueHeadsPerGroup,
			keyHeads, tensor.PairedExtent*valueHeadsPerGroup,
		)
		beta = builder.Reshape(builder.Sigmoid(beta), tensor.SingletonExtent, valueHeads, tokens, sequences)
		alpha = builder.Reshape(alpha, valueHeads, tokens, sequences)
	} else {
		beta = builder.Reshape(
			builder.Sigmoid(builder.MulMat(weights.SSMBeta, normalized)),
			tensor.SingletonExtent, valueHeads, tokens, sequences,
		)
		alpha = builder.MulMat(weights.SSMAlpha, normalized)
		if sequences > tensor.SingletonExtent {
			alpha = builder.Reshape(alpha, valueHeads, tokens, sequences)
		}
	}
	gate := builder.Multiply(
		builder.Softplus(builder.Add(alpha, weights.SSMTimeStep)),
		weights.SSMA,
	)
	gate = builder.Reshape(gate, tensor.SingletonExtent, valueHeads, tokens, sequences)

	var qkvTime *tensor.Tensor
	if sequences > tensor.SingletonExtent && tokens == tensor.SingletonExtent {
		qkvTime = builder.Reshape(qkvMixed, tokens, convChannels, sequences)
	} else {
		qkvTime = builder.Transpose2D(qkvMixed)
		if sequences > tensor.SingletonExtent {
			qkvTime = builder.Reshape(qkvTime, tokens, convChannels, sequences)
		}
	}
	convInput := builder.Concat(convState, qkvTime, tensor.FirstOffset)
	nextConvState := builder.GroupSlice(
		convInput,
		tokens,
		uint64(spec.SSMConvKernel-tensor.SingletonExtent),
		tensor.SingletonExtent,
		uint64(spec.SSMConvKernel-tensor.SingletonExtent),
	)
	if sequences == tensor.SingletonExtent {
		nextConvState = builder.Reshape(
			nextConvState, uint64(spec.SSMConvKernel-tensor.SingletonExtent), convChannels,
		)
	} else {
		nextConvState = builder.Reshape(
			nextConvState, uint64(spec.SSMConvKernel-tensor.SingletonExtent), convChannels, sequences,
		)
	}
	convolved := builder.SiLU(builder.SSMConv(convInput, weights.SSMConv1D))
	query := builder.GroupSlice(convolved, tensor.FirstOffset, stateWidth, keyHeads, stateWidth)
	key := builder.GroupSlice(convolved, keyDimension, stateWidth, keyHeads, stateWidth)
	value := builder.GroupSlice(convolved, tensor.PairedExtent*keyDimension, stateWidth, valueHeads, stateWidth)
	query = builder.Reshape(builder.L2Norm(query, spec.RMSNormEpsilon), stateWidth, keyHeads, tokens, sequences)
	key = builder.Reshape(builder.L2Norm(key, spec.RMSNormEpsilon), stateWidth, keyHeads, tokens, sequences)
	value = builder.Reshape(value, stateWidth, valueHeads, tokens, sequences)
	var packed *tensor.Tensor
	if deltaPolicy == gatedDeltaInterleavedProjections {
		packed = builder.GatedDeltaNetRepeatInterleave(query, key, value, gate, beta, ssmState)
	} else {
		packed = builder.GatedDeltaNet(query, key, value, gate, beta, ssmState)
	}
	attentionElements := stateWidth * valueHeads * tokens * sequences
	attention := builder.FlatSlice(
		packed,
		tensor.FirstOffset,
		stateWidth,
		valueHeads,
		tokens,
		sequences,
	)
	nextSSMState := builder.FlatSlice(
		packed,
		attentionElements,
		stateWidth,
		stateWidth,
		valueHeads,
		sequences,
	)
	z = builder.Reshape(z, stateWidth, valueHeads, tokens, sequences)
	attention = builder.Multiply(
		builder.WeightedRMSNorm(attention, weights.SSMNorm, spec.RMSNormEpsilon),
		builder.SiLU(z),
	)
	attention = builder.Reshape(attention, valueDimension, tokens*sequences)
	attention = builder.MulMat(weights.SSMOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextConvState, Value: nextSSMState}, nil
}

func buildRoutedSwiGLUFeedForwardMix(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	weights LayerGraphWeights,
	experts MoEGraphPlan,
	composition ExpertCompositionPlan,
) (*tensor.Tensor, error) {
	required := graphWeights{}
	addGatedDeltaFeedForwardRequirements(&required, composition, weights)
	if err := required.validate("gated-delta feed-forward mix"); err != nil {
		return nil, err
	}
	var feedForward *tensor.Tensor
	if composition.kind == expertSharedGated {
		feedForward = experts.BuildLayer(builder, normalized, nil, weights)
		sharedRouter := builder.Reshape(
			weights.FeedForwardSharedRouter, normalized.Shape.ContiguousExtent(), tensor.SingletonExtent,
		)
		sharedGate := builder.Sigmoid(builder.MulMat(sharedRouter, normalized))
		shared := builder.MulMat(
			weights.FeedForwardSharedDown,
			builder.SwiGLU(
				builder.MulMat(weights.FeedForwardSharedGate, normalized),
				builder.MulMat(weights.FeedForwardSharedUp, normalized),
			),
		)
		feedForward = builder.Add(feedForward, builder.Multiply(shared, sharedGate))
	} else {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	}
	return feedForward, builder.Err()
}

func addGatedDeltaFeedForwardRequirements(
	required *graphWeights,
	composition ExpertCompositionPlan,
	weights LayerGraphWeights,
) {
	if composition.kind == expertSharedGated {
		*required = append(*required, weights.FeedForwardRouter)
		*required = append(*required, weights.FeedForwardDownExperts)
		if weights.FeedForwardGateUpExperts != nil {
			*required = append(*required, weights.FeedForwardGateUpExperts)
		} else {
			*required = append(*required, weights.FeedForwardGateExperts)
			*required = append(*required, weights.FeedForwardUpExperts)
		}
		*required = append(*required, weights.FeedForwardSharedRouter)
		*required = append(*required, weights.FeedForwardSharedGate)
		*required = append(*required, weights.FeedForwardSharedUp)
		*required = append(*required, weights.FeedForwardSharedDown)
		return
	}
	*required = append(*required, weights.FeedForwardGate)
	*required = append(*required, weights.FeedForwardUp)
	*required = append(*required, weights.FeedForwardDown)
}
