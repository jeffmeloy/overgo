package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// Gemma4AssistantSession: target KV plus recurrent hidden row.
type Gemma4AssistantSession struct {
	TargetCache   *KVCache
	PendingHidden reference.Value
	Position      uint32
}

// NewGemma4AssistantSession: target-prefix shared-context setup.
func (r *Runner) NewGemma4AssistantSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*Gemma4AssistantSession, error) {
	return r.newGemma4AssistantSession(ctx, target, tokenIDs, nil)
}

// NewGemma4AssistantProjectedSession: media-aware target-prefix setup.
func (r *Runner) NewGemma4AssistantProjectedSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
	inputs ProjectedInputs,
) (*Gemma4AssistantSession, error) {
	return r.newGemma4AssistantSession(ctx, target, tokenIDs, &inputs)
}

func (r *Runner) newGemma4AssistantSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
	inputs *ProjectedInputs,
) (*Gemma4AssistantSession, error) {
	if r == nil || target == nil || r == target || r.path == target.path || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Gemma 4 assistant and target inputs are invalid")
	}
	if err := r.validateGemma4AssistantTarget(target); err != nil {
		return nil, err
	}
	var hidden reference.Value
	var cache *KVCache
	var err error
	if inputs == nil {
		hidden, cache, err = target.ForwardCached(ctx, tokenIDs, nil)
	} else {
		hidden, cache, err = target.ForwardCachedWithProjectedInputs(ctx, tokenIDs, nil, *inputs)
	}
	if err != nil {
		return nil, err
	}
	last := lastHiddenColumn(hidden)
	return &Gemma4AssistantSession{
		TargetCache: cache, PendingHidden: last, Position: effectiveCachePosition(cache),
	}, nil
}

// AdvanceGemma4Assistant: one fixed-position draft step.
func (r *Runner) AdvanceGemma4Assistant(
	ctx context.Context,
	target *Runner,
	tokenID tokenizer.TokenID,
	session *Gemma4AssistantSession,
) (reference.Value, *Gemma4AssistantSession, error) {
	if r == nil || target == nil || session == nil || session.TargetCache == nil {
		return reference.Value{}, nil, errors.New("inference: Gemma 4 assistant session is invalid")
	}
	first, second := r, target
	if first.path > second.path {
		first, second = second, first
	}
	first.mu.Lock()
	second.mu.Lock()
	defer second.mu.Unlock()
	defer first.mu.Unlock()
	if r.closed || target.closed {
		return reference.Value{}, nil, errors.New("inference: Gemma 4 assistant runner is unavailable")
	}
	if err := r.validateGemma4AssistantTarget(target); err != nil {
		return reference.Value{}, nil, err
	}
	if err := target.validateCache(session.TargetCache); err != nil {
		return reference.Value{}, nil, fmt.Errorf("inference: Gemma 4 target cache: %w", err)
	}
	if effectiveCachePosition(session.TargetCache) != session.Position ||
		session.PendingHidden.Shape.Rank != 2 ||
		session.PendingHidden.Shape.Dims[0] != uint64(r.spec.TargetHiddenSize) ||
		session.PendingHidden.Shape.Dims[1] != 1 {
		return reference.Value{}, nil, errors.New("inference: Gemma 4 assistant session state is incompatible")
	}
	if tokenID < 0 || int(tokenID) >= target.vocab.Len() {
		return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	targetEmbedding, err := target.loadEmbeddings(ctx, []uint32{uint32(tokenID)})
	if err != nil {
		return reference.Value{}, nil, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	tokenInput := runtime.input("gemma4_assistant.target_token", targetEmbedding)
	hiddenInput := runtime.input("gemma4_assistant.target_hidden", session.PendingHidden)
	pre, err := runtime.weight(*r.weights.FeatureProjection)
	if err != nil {
		return reference.Value{}, nil, err
	}
	current, err := model.BuildGemma4AssistantInput(runtime.builder, tokenInput, hiddenInput, pre, r.spec)
	if err != nil {
		return reference.Value{}, nil, err
	}
	cacheInputs := make(map[bool][2]*tensor.Tensor, 2)
	for _, sliding := range []bool{true, false} {
		source := len(session.TargetCache.Layers) - 1
		if sliding {
			source--
		}
		layerCache := session.TargetCache.Layers[source]
		key := runtime.input(fmt.Sprintf("gemma4_assistant.shared_%t_key", sliding), layerCache.Key)
		value := runtime.input(fmt.Sprintf("gemma4_assistant.shared_%t_value", sliding), layerCache.Value)
		cacheInputs[sliding] = [2]*tensor.Tensor{key, value}
	}
	for layerIndex, info := range r.weights.Layers {
		graphWeights, layerErr := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
		if layerErr != nil {
			return reference.Value{}, nil, layerErr
		}
		shared := cacheInputs[r.spec.IsSlidingLayer(uint32(layerIndex))]
		current, err = model.BuildGemma4AssistantBlockWithPlan(
			runtime.builder, current, r.spec, graphWeights, []uint32{session.Position},
			shared[0], shared[1], r.layerPlan(layerIndex),
		)
		if err != nil {
			return reference.Value{}, nil, fmt.Errorf("inference Gemma 4 assistant layer %d: %w", layerIndex, err)
		}
	}
	outputNorm, err := runtime.weight(r.weights.OutputNorm)
	if err != nil {
		return reference.Value{}, nil, err
	}
	output, err := runtime.weight(r.weights.TokenEmbedding)
	if err != nil {
		return reference.Value{}, nil, err
	}
	post, err := runtime.weight(*r.weights.FeatureProjectionPost)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, nextHidden, err := model.BuildGemma4AssistantOutputs(runtime.builder, current, outputNorm, output, post, r.spec)
	if err != nil {
		return reference.Value{}, nil, err
	}
	results, err := runtime.execute(logits, nextHidden)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &Gemma4AssistantSession{
		TargetCache: session.TargetCache, PendingHidden: results[nextHidden], Position: session.Position,
	}
	return results[logits], next, nil
}

func (r *Runner) validateGemma4AssistantTarget(target *Runner) error {
	if r.profile().Forward != model.ForwardGemma4Assistant || target.profile().DenseGraph != model.DenseGraphGemma4 ||
		target.spec.EmbeddingLength != r.spec.TargetHiddenSize ||
		target.spec.VocabularySize != r.spec.VocabularySize || target.spec.BlockCount < 2 ||
		!target.spec.IsSlidingLayer(target.spec.BlockCount-2) ||
		target.spec.IsSlidingLayer(target.spec.BlockCount-1) {
		return errors.New("inference: Gemma 4 assistant target model is incompatible")
	}
	return nil
}
