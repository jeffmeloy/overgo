package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// FeatureDraftSession: shifted feature plus draft KV.
type FeatureDraftSession struct {
	Cache          *KVCache
	TargetCache    *KVCache
	TargetTokens   []tokenizer.TokenID
	PendingFeature reference.Value
	Position       uint32
}

// featureDraftStep: logits, pre-norm feature, and cache.
type featureDraftStep struct {
	Logits      reference.Value
	NextFeature reference.Value
	Cache       *KVCache
}

// NewFeatureDraftSession: full-prefix shifted-cache construction.
func (r *Runner) NewFeatureDraftSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*FeatureDraftSession, error) {
	if r == nil || target == nil || r == target || r.path == target.path || !checked.Nonzero(len(tokenIDs)) {
		return nil, errors.New("inference: feature-draft inputs are invalid")
	}
	if r.forwardProgram().Session != model.ForwardSessionFeatureDraft || target.spec.EmbeddingLength != r.spec.TargetHiddenSize {
		return nil, errors.New("inference: feature-draft target is incompatible")
	}
	_, targetCache, features, err := target.ForwardCachedExtractLayerInputs(
		ctx, tokenIDs, nil, r.spec.TargetLayers,
	)
	if err != nil {
		return nil, err
	}
	fused, err := r.projectFeatures(ctx, features)
	if err != nil {
		return nil, err
	}
	_, _, validFused := fused.MatrixExtents()
	if !validFused {
		return nil, errors.New("inference: feature-draft projection shape is incompatible")
	}
	pending, err := reference.FinalRows(fused, tensor.SingletonExtent)
	if err != nil {
		return nil, err
	}
	prefix, _ := checked.Init(tokenIDs)
	tail, _ := checked.Tail(tokenIDs)
	session := &FeatureDraftSession{
		TargetCache: targetCache, TargetTokens: slices.Clone(tokenIDs),
		PendingFeature: pending, Position: uint32(len(prefix)),
	}
	for index := range prefix {
		feature, selectErr := reference.SelectRows(fused, uint64(index), tensor.SingletonExtent)
		if selectErr != nil {
			return nil, selectErr
		}
		step, stepErr := r.stepFeatureDraft(ctx, target, tail[index], feature, uint32(index), session.Cache)
		if stepErr != nil {
			return nil, stepErr
		}
		session.Cache = step.Cache
	}
	return session, nil
}

// AdvanceFeatureDraft: one autoregressive draft step.
func (r *Runner) AdvanceFeatureDraft(
	ctx context.Context,
	target *Runner,
	tokenID tokenizer.TokenID,
	session *FeatureDraftSession,
) (reference.Value, *FeatureDraftSession, error) {
	if session == nil {
		return reference.Value{}, nil, errors.New("inference: feature-draft session is invalid")
	}
	if err := r.spec.ValidateSequenceRow(session.PendingFeature); err != nil {
		return reference.Value{}, nil, errors.New("inference: feature-draft session is invalid")
	}
	step, err := r.stepFeatureDraft(ctx, target, tokenID, session.PendingFeature, session.Position, session.Cache)
	if err != nil {
		return reference.Value{}, nil, err
	}
	nextPosition := session.Position
	nextPosition++
	next := &FeatureDraftSession{
		Cache: step.Cache, TargetCache: session.TargetCache,
		TargetTokens:   slices.Clone(session.TargetTokens),
		PendingFeature: step.NextFeature, Position: nextPosition,
	}
	return step.Logits, next, nil
}

func (r *Runner) stepFeatureDraft(
	ctx context.Context,
	target *Runner,
	tokenID tokenizer.TokenID,
	feature reference.Value,
	position uint32,
	cache *KVCache,
) (featureDraftStep, error) {
	if r == nil || target == nil || r == target || r.path == target.path {
		return featureDraftStep{}, errors.New("inference: feature-draft runners are invalid")
	}
	first, second := r, target
	if first.path > second.path {
		first, second = second, first
	}
	first.mu.Lock()
	second.mu.Lock()
	defer second.mu.Unlock()
	defer first.mu.Unlock()
	if r.closed || target.closed || r.forwardProgram().Session != model.ForwardSessionFeatureDraft {
		return featureDraftStep{}, errors.New("inference: feature-draft runner is unavailable")
	}
	if !checked.NonNegativeInts(int(tokenID)) || int(tokenID) >= target.vocab.Len() {
		return featureDraftStep{}, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	if err := r.spec.ValidateSequenceRow(feature); err != nil {
		return featureDraftStep{}, errors.New("inference: feature-draft input shape is incompatible")
	}
	if cache != nil && (len(cache.Layers) != tensor.SingletonExtent || cache.Position != position) {
		return featureDraftStep{}, errors.New("inference: feature-draft cache position is incompatible")
	}
	rows := []uint32{uint32(tokenID)}
	tokenEmbedding, err := r.gatherTensor(ctx, r.weights.TokenEmbedding, rows)
	if err != nil {
		return featureDraftStep{}, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	tokenInput := runtime.input("feature_draft.token", tokenEmbedding)
	featureInput := runtime.input("feature_draft.feature", feature)
	graphWeights, err := runtime.layer(r.weights.Layers[tensor.FirstOffset], "blk.0.")
	if err != nil {
		return featureDraftStep{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if cache != nil {
		pastKey = runtime.input("feature_draft.past_key", cache.Layers[tensor.FirstOffset].Key)
		pastValue = runtime.input("feature_draft.past_value", cache.Layers[tensor.FirstOffset].Value)
	}
	program := r.layerProgram(tensor.FirstOffset)
	plan := program.Layer()
	block, err := program.Build(model.CachedBlockContext{
		Builder: runtime.builder, Input: tokenInput, Positions: []uint32{position},
		PastKey: pastKey, PastValue: pastValue, PerLayerInput: featureInput,
		Layer: plan.Layer, CacheWrite: tensor.CacheWriteConcat,
	}, graphWeights)
	if err != nil {
		return featureDraftStep{}, err
	}
	norm, err := runtime.weight(r.weights.OutputNorm)
	if err != nil {
		return featureDraftStep{}, err
	}
	normalized := runtime.builder.WeightedRMSNorm(block.Output, norm, r.spec.RMSNormEpsilon)
	outputInfo := target.outputTensor()
	outputOwner := target
	if r.program.Model.Terminal().OutputHead == model.OutputHeadDedicated {
		outputInfo, outputOwner = r.outputTensor(), r
	}
	var output *tensor.Tensor
	if outputOwner == r {
		output, err = runtime.weight(outputInfo)
	} else {
		var value reference.Value
		value, err = target.hostTensor(ctx, outputInfo)
		if err == nil {
			output = runtime.input(outputInfo.Name+".shared", value)
		}
	}
	if err != nil {
		return featureDraftStep{}, err
	}
	logits := runtime.builder.MulMat(output, normalized)
	outputs := []*tensor.Tensor{block.Output, block.Key, block.Value, logits}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return featureDraftStep{}, err
	}
	logitValue := results[logits]
	if r.weights.DraftToTarget != nil {
		logitValue, err = r.remapFeatureDraftLogits(ctx, logitValue)
		if err != nil {
			return featureDraftStep{}, err
		}
	}
	_, _, cacheTokens, validCache := tensor.Extents3(results[block.Key].Shape)
	if !validCache {
		return featureDraftStep{}, errors.New("inference: feature-draft cache shape is incompatible")
	}
	nextPosition := position
	nextPosition++
	nextCache := &KVCache{
		Layers: []LayerCache{{Key: results[block.Key], Value: results[block.Value]}},
		Tokens: uint32(cacheTokens), Position: nextPosition,
	}
	return featureDraftStep{Logits: logitValue, NextFeature: results[block.Output], Cache: nextCache}, nil
}

func (r *Runner) remapFeatureDraftLogits(ctx context.Context, logits reference.Value) (reference.Value, error) {
	mapping, err := r.hostTensor(ctx, *r.weights.DraftToTarget)
	if err != nil {
		return reference.Value{}, err
	}
	result := reference.Value{Shape: tensor.MustShape(uint64(r.spec.VocabularySize), tensor.SingletonExtent), Data: make([]float32, r.spec.VocabularySize)}
	for index := range result.Data {
		result.Data[index] = float32(math.Inf(-tensor.SingletonExtent))
	}
	for draft, rawTarget := range mapping.Data {
		target := int(rawTarget)
		if !checked.NonNegativeInts(target) || target >= len(result.Data) || draft >= len(logits.Data) {
			return reference.Value{}, errors.New("inference: feature-draft vocabulary map is invalid")
		}
		result.Data[target] = logits.Data[draft]
	}
	return result, nil
}
