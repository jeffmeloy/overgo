package inference

import (
	"context"
	"errors"
	"fmt"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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
	if r == nil || target == nil || r == target || r.path == target.path || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Gemma 4 assistant and target inputs are invalid")
	}
	if err := r.validateGemma4AssistantTarget(target); err != nil {
		return nil, err
	}
	hidden, cache, err := target.ForwardCached(ctx, tokenIDs, nil)
	if err != nil {
		return nil, err
	}
	width := int(hidden.Shape.Dims[0])
	last := reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), hidden.Data[len(hidden.Data)-width:]...),
	}
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
	builder := r.newGraphBuilder()
	tokenInput := builder.Input("gemma4_assistant.target_token", dtype.F32, targetEmbedding.Shape)
	hiddenInput := builder.Input("gemma4_assistant.target_hidden", dtype.F32, session.PendingHidden.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{
		tokenInput: targetEmbedding, hiddenInput: session.PendingHidden,
	}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	pre, pointer, err := r.deviceOrHostTensor(ctx, builder, *r.weights.FeatureProjection, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[pre] = pointer
	}
	current, err := model.BuildGemma4AssistantInput(builder, tokenInput, hiddenInput, pre, r.spec)
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
		key := builder.Input(fmt.Sprintf("gemma4_assistant.shared_%t_key", sliding), dtype.F32, layerCache.Key.Shape)
		value := builder.Input(fmt.Sprintf("gemma4_assistant.shared_%t_value", sliding), dtype.F32, layerCache.Value.Shape)
		hostFeeds[key], hostFeeds[value] = layerCache.Key, layerCache.Value
		cacheInputs[sliding] = [2]*tensor.Tensor{key, value}
	}
	for layerIndex, info := range r.weights.Layers {
		graphWeights, layerDeviceFeeds, layerErr := r.gemma4AssistantLayerInputs(ctx, builder, hostFeeds, info, layerIndex)
		if layerErr != nil {
			return reference.Value{}, nil, layerErr
		}
		for node, layerPointer := range layerDeviceFeeds {
			deviceFeeds[node] = layerPointer
		}
		shared := cacheInputs[r.spec.IsSlidingLayer(uint32(layerIndex))]
		current, err = model.BuildGemma4AssistantBlock(
			builder, current, r.spec, graphWeights, []uint32{session.Position},
			shared[0], shared[1], uint32(layerIndex),
		)
		if err != nil {
			return reference.Value{}, nil, fmt.Errorf("inference Gemma 4 assistant layer %d: %w", layerIndex, err)
		}
	}
	outputNorm, pointer, err := r.deviceOrHostTensor(ctx, builder, r.weights.OutputNorm, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[outputNorm] = pointer
	}
	output, pointer, err := r.deviceOrHostTensor(ctx, builder, r.weights.TokenEmbedding, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[output] = pointer
	}
	post, pointer, err := r.deviceOrHostTensor(ctx, builder, *r.weights.FeatureProjectionPost, hostFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if pointer != 0 {
		deviceFeeds[post] = pointer
	}
	logits, nextHidden, err := model.BuildGemma4AssistantOutputs(builder, current, outputNorm, output, post, r.spec)
	if err != nil {
		return reference.Value{}, nil, err
	}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{logits, nextHidden}, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, []*tensor.Tensor{logits, nextHidden}, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &Gemma4AssistantSession{
		TargetCache: session.TargetCache, PendingHidden: results[nextHidden], Position: session.Position,
	}
	return results[logits], next, nil
}

func (r *Runner) validateGemma4AssistantTarget(target *Runner) error {
	if r.spec.Architecture != "gemma4-assistant" || target.spec.Architecture != "gemma4" ||
		target.spec.EmbeddingLength != r.spec.TargetHiddenSize ||
		target.spec.VocabularySize != r.spec.VocabularySize || target.spec.BlockCount < 2 ||
		!target.spec.IsSlidingLayer(target.spec.BlockCount-2) ||
		target.spec.IsSlidingLayer(target.spec.BlockCount-1) {
		return errors.New("inference: Gemma 4 assistant target model is incompatible")
	}
	return nil
}

func (r *Runner) gemma4AssistantLayerInputs(
	ctx context.Context,
	builder *tensor.Builder,
	hostFeeds map[*tensor.Tensor]reference.Value,
	info model.LayerWeights,
	layerIndex int,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	if r.hasPreloadedWeights() {
		return r.layerDeviceInputs(builder, info)
	}
	hostLayer, err := model.LoadHostLayer(ctx, r.file, info)
	if err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	graph, feeds, err := hostLayer.GraphInputs(builder, fmt.Sprintf("blk.%d.", layerIndex))
	if err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	for node, value := range feeds {
		hostFeeds[node] = value
	}
	return graph, map[*tensor.Tensor]driver.DevicePtr{}, nil
}
