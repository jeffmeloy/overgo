package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// Eagle3Session: shifted feature plus draft KV.
type Eagle3Session struct {
	Cache          *KVCache
	TargetCache    *KVCache
	TargetTokens   []tokenizer.TokenID
	PendingFeature reference.Value
	Position       uint32
}

// Eagle3StepResult: logits, pre-norm feature, and cache.
type Eagle3StepResult struct {
	Logits      reference.Value
	NextFeature reference.Value
	Cache       *KVCache
}

// FuseEagle3Features: projects three target-layer inputs.
func (r *Runner) FuseEagle3Features(ctx context.Context, features reference.Value) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.profile().Forward != model.ForwardEagle3 || r.weights.FeatureProjection == nil {
		return reference.Value{}, errors.New("inference: Eagle3 feature encoder is unavailable")
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("eagle3.features", features)
	projection, err := runtime.weight(*r.weights.FeatureProjection)
	if err != nil {
		return reference.Value{}, err
	}
	output, err := model.BuildEagle3FeatureEncoder(runtime.builder, input, projection, r.spec)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

// NewEagle3Session: full-prefix shifted-cache construction.
func (r *Runner) NewEagle3Session(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*Eagle3Session, error) {
	if r == nil || target == nil || r == target || r.path == target.path || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Eagle3 and target inputs are invalid")
	}
	if r.profile().Forward != model.ForwardEagle3 || target.spec.EmbeddingLength != r.spec.TargetHiddenSize {
		return nil, errors.New("inference: Eagle3 target model is incompatible")
	}
	_, targetCache, features, err := target.ForwardCachedExtractLayerInputs(
		ctx, tokenIDs, nil, r.spec.TargetLayers,
	)
	if err != nil {
		return nil, err
	}
	fused, err := r.FuseEagle3Features(ctx, features)
	if err != nil {
		return nil, err
	}
	width := int(fused.Shape.Dims[0])
	pending := reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  slices.Clone(fused.Data[(len(tokenIDs)-1)*width:]),
	}
	session := &Eagle3Session{
		TargetCache: targetCache, TargetTokens: slices.Clone(tokenIDs),
		PendingFeature: pending, Position: uint32(len(tokenIDs) - 1),
	}
	for index := 0; index+1 < len(tokenIDs); index++ {
		feature := reference.Value{
			Shape: tensor.MustShape(uint64(width), 1),
			Data:  slices.Clone(fused.Data[index*width : (index+1)*width]),
		}
		step, stepErr := r.stepEagle3(ctx, target, tokenIDs[index+1], feature, uint32(index), session.Cache)
		if stepErr != nil {
			return nil, stepErr
		}
		session.Cache = step.Cache
	}
	return session, nil
}

// AdvanceEagle3: one autoregressive draft step.
func (r *Runner) AdvanceEagle3(
	ctx context.Context,
	target *Runner,
	tokenID tokenizer.TokenID,
	session *Eagle3Session,
) (reference.Value, *Eagle3Session, error) {
	if session == nil || session.PendingFeature.Shape.Rank != 2 {
		return reference.Value{}, nil, errors.New("inference: Eagle3 session is invalid")
	}
	step, err := r.stepEagle3(ctx, target, tokenID, session.PendingFeature, session.Position, session.Cache)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &Eagle3Session{
		Cache: step.Cache, TargetCache: session.TargetCache,
		TargetTokens:   slices.Clone(session.TargetTokens),
		PendingFeature: step.NextFeature, Position: session.Position + 1,
	}
	return step.Logits, next, nil
}

func (r *Runner) stepEagle3(
	ctx context.Context,
	target *Runner,
	tokenID tokenizer.TokenID,
	feature reference.Value,
	position uint32,
	cache *KVCache,
) (Eagle3StepResult, error) {
	if r == nil || target == nil || r == target || r.path == target.path {
		return Eagle3StepResult{}, errors.New("inference: Eagle3 and target runners are invalid")
	}
	first, second := r, target
	if first.path > second.path {
		first, second = second, first
	}
	first.mu.Lock()
	second.mu.Lock()
	defer second.mu.Unlock()
	defer first.mu.Unlock()
	if r.closed || target.closed || r.profile().Forward != model.ForwardEagle3 {
		return Eagle3StepResult{}, errors.New("inference: Eagle3 runner is unavailable")
	}
	if tokenID < 0 || int(tokenID) >= target.vocab.Len() {
		return Eagle3StepResult{}, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	if feature.Shape.Rank != 2 || feature.Shape.Dims[0] != uint64(r.spec.EmbeddingLength) || feature.Shape.Dims[1] != 1 {
		return Eagle3StepResult{}, errors.New("inference: Eagle3 feature shape is incompatible")
	}
	if cache != nil && (len(cache.Layers) != 1 || cache.Position != position) {
		return Eagle3StepResult{}, errors.New("inference: Eagle3 cache position is incompatible")
	}
	rows := []uint32{uint32(tokenID)}
	var tokenEmbedding reference.Value
	var err error
	if r.weights.TokenEmbedding.Name != "" {
		tokenEmbedding, err = r.loadRows(ctx, r.weights.TokenEmbedding, rows)
	} else {
		tokenEmbedding, err = target.loadEmbeddings(ctx, rows)
	}
	if err != nil {
		return Eagle3StepResult{}, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	tokenInput := runtime.input("eagle3.token", tokenEmbedding)
	featureInput := runtime.input("eagle3.feature", feature)
	graphWeights, err := runtime.layer(r.weights.Layers[0], "blk.0.")
	if err != nil {
		return Eagle3StepResult{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if cache != nil {
		pastKey = runtime.input("eagle3.past_key", cache.Layers[0].Key)
		pastValue = runtime.input("eagle3.past_value", cache.Layers[0].Value)
	}
	program := r.layerProgram(0)
	plan := program.Layer()
	block, err := program.Build(model.CachedBlockContext{
		Builder: runtime.builder, Input: tokenInput, Positions: []uint32{position},
		PastKey: pastKey, PastValue: pastValue, PerLayerInput: featureInput,
		Layer: plan.Layer, CacheWrite: tensor.CacheWriteConcat,
	}, graphWeights)
	if err != nil {
		return Eagle3StepResult{}, err
	}
	norm, err := runtime.weight(r.weights.OutputNorm)
	if err != nil {
		return Eagle3StepResult{}, err
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
		return Eagle3StepResult{}, err
	}
	logits := runtime.builder.MulMat(output, normalized)
	outputs := []*tensor.Tensor{block.Output, block.Key, block.Value, logits}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return Eagle3StepResult{}, err
	}
	logitValue := results[logits]
	if r.weights.DraftToTarget != nil {
		logitValue, err = r.remapEagle3Logits(ctx, logitValue)
		if err != nil {
			return Eagle3StepResult{}, err
		}
	}
	nextCache := &KVCache{
		Layers: []LayerCache{{Key: results[block.Key], Value: results[block.Value]}},
		Tokens: uint32(results[block.Key].Shape.Dims[2]), Position: position + 1,
	}
	return Eagle3StepResult{Logits: logitValue, NextFeature: results[block.Output], Cache: nextCache}, nil
}

func (r *Runner) remapEagle3Logits(ctx context.Context, logits reference.Value) (reference.Value, error) {
	mapping, err := r.hostTensor(ctx, *r.weights.DraftToTarget)
	if err != nil {
		return reference.Value{}, err
	}
	result := reference.Value{Shape: tensor.MustShape(uint64(r.spec.VocabularySize), 1), Data: make([]float32, r.spec.VocabularySize)}
	for index := range result.Data {
		result.Data[index] = float32(math.Inf(-1))
	}
	for draft, rawTarget := range mapping.Data {
		target := int(rawTarget)
		if target < 0 || target >= len(result.Data) || draft >= len(logits.Data) {
			return reference.Value{}, errors.New("inference: Eagle3 vocabulary map is invalid")
		}
		result.Data[target] = logits.Data[draft]
	}
	return result, nil
}
