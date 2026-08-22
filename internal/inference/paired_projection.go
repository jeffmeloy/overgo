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

// PairedProjectionSession: target KV plus recurrent hidden row.
type PairedProjectionSession struct {
	TargetCache   *KVCache
	PendingHidden reference.Value
	Position      uint32
}

// NewPairedProjectionSession creates shared target-prefix context.
func (r *Runner) NewPairedProjectionSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
	inputs *ProjectedInputs,
) (*PairedProjectionSession, error) {
	if r == nil || target == nil || r == target || r.path == target.path || len(tokenIDs) == 0 {
		return nil, errors.New("inference: paired projection inputs are invalid")
	}
	if err := r.validatePairedProjectionTarget(target); err != nil {
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
	last := hidden.LastRowView().Clone()
	return &PairedProjectionSession{
		TargetCache: cache, PendingHidden: last, Position: effectiveCachePosition(cache),
	}, nil
}

// AdvancePairedProjection executes one fixed-position draft step.
func (r *Runner) AdvancePairedProjection(
	ctx context.Context,
	target *Runner,
	tokenID tokenizer.TokenID,
	session *PairedProjectionSession,
) (reference.Value, *PairedProjectionSession, error) {
	if r == nil || target == nil || session == nil || session.TargetCache == nil {
		return reference.Value{}, nil, errors.New("inference: paired projection session is invalid")
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
		return reference.Value{}, nil, errors.New("inference: paired projection runner is unavailable")
	}
	if err := r.validatePairedProjectionTarget(target); err != nil {
		return reference.Value{}, nil, err
	}
	if err := target.validateCache(session.TargetCache); err != nil {
		return reference.Value{}, nil, fmt.Errorf("inference: paired target cache: %w", err)
	}
	if effectiveCachePosition(session.TargetCache) != session.Position ||
		session.PendingHidden.Shape.Rank != 2 ||
		session.PendingHidden.Shape.Dims[0] != uint64(r.spec.TargetHiddenSize) ||
		session.PendingHidden.Shape.Dims[1] != 1 {
		return reference.Value{}, nil, errors.New("inference: paired projection session state is incompatible")
	}
	if tokenID < 0 || int(tokenID) >= target.vocab.Len() {
		return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	targetEmbedding, err := target.gatherTensor(ctx, target.weights.TokenEmbedding, []uint32{uint32(tokenID)})
	if err != nil {
		return reference.Value{}, nil, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	tokenInput := runtime.input("paired_projection.target_token", targetEmbedding)
	hiddenInput := runtime.input("paired_projection.target_hidden", session.PendingHidden)
	pre, err := runtime.weight(*r.weights.FeatureProjection)
	if err != nil {
		return reference.Value{}, nil, err
	}
	fused, err := r.program.Model.Projection(model.ProjectionPairedInput).Build(
		runtime.builder,
		model.ProjectionOperands{Input: tokenInput, Paired: hiddenInput, Primary: pre},
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	current := fused.Primary
	cacheInputs := make(map[bool][2]*tensor.Tensor, 2)
	for _, sliding := range []bool{true, false} {
		source := len(session.TargetCache.Layers) - 1
		if sliding {
			source--
		}
		layerCache := session.TargetCache.Layers[source]
		key := runtime.input(fmt.Sprintf("paired_projection.shared_%t_key", sliding), layerCache.Key)
		value := runtime.input(fmt.Sprintf("paired_projection.shared_%t_value", sliding), layerCache.Value)
		cacheInputs[sliding] = [2]*tensor.Tensor{key, value}
	}
	for layerIndex, info := range r.weights.Layers {
		program := r.layerProgram(layerIndex)
		plan := program.Layer()
		graphWeights, layerErr := runtime.layer(info, fmt.Sprintf("blk.%d.", layerIndex))
		if layerErr != nil {
			return reference.Value{}, nil, layerErr
		}
		shared := cacheInputs[plan.Sliding]
		block, err := program.Build(model.CachedBlockContext{
			Builder: runtime.builder, Input: current, Positions: []uint32{session.Position},
			PastKey: shared[0], PastValue: shared[1], Layer: plan.Layer,
		}, graphWeights)
		if err != nil {
			return reference.Value{}, nil, fmt.Errorf("inference paired projection layer %d: %w", layerIndex, err)
		}
		current = block.Output
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
	projected, err := r.program.Model.Projection(model.ProjectionPairedOutput).Build(
		runtime.builder,
		model.ProjectionOperands{
			Input: current, Primary: output, Secondary: post, Normalization: outputNorm,
		},
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, nextHidden := projected.Primary, projected.Secondary
	results, err := runtime.execute(logits, nextHidden)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &PairedProjectionSession{
		TargetCache: session.TargetCache, PendingHidden: results[nextHidden], Position: session.Position,
	}
	return results[logits], next, nil
}

func (r *Runner) validatePairedProjectionTarget(target *Runner) error {
	if target.spec.BlockCount < tensor.PairedExtent {
		return errors.New("inference: paired projection target model is incompatible")
	}
	penultimate, penultimateErr := target.program.Model.Layer(target.spec.BlockCount - tensor.PairedExtent)
	last, lastErr := target.program.Model.Layer(target.spec.BlockCount - tensor.SingletonExtent)
	if r.forwardProgram().Session != model.ForwardSessionPairedProjection ||
		target.forwardProgram().Operation != model.ForwardOperationCached ||
		target.spec.EmbeddingLength != r.spec.TargetHiddenSize ||
		target.spec.VocabularySize != r.spec.VocabularySize ||
		penultimateErr != nil || lastErr != nil || !penultimate.Sliding || last.Sliding {
		return errors.New("inference: paired projection target model is incompatible")
	}
	return nil
}
