package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

func (target *Runner) newSingleHeadMTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*Qwen35MTPSession, error) {
	hidden, cache, err := target.ForwardCached(ctx, tokenIDs, nil)
	if err != nil {
		return nil, err
	}
	last := lastHiddenColumn(hidden)
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	position := effectiveCachePosition(cache)
	return &Qwen35MTPSession{
		TrunkCache: cache, PendingHidden: last, MTPStart: position, Position: position,
		targetModel: targetModel,
	}, nil
}

func lastHiddenColumn(hidden reference.Value) reference.Value {
	width := int(hidden.Shape.Dims[0])
	return reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), hidden.Data[len(hidden.Data)-width:]...),
	}
}

func (r *Runner) validateSingleHeadMTPSession(
	session *Qwen35MTPSession,
	label string,
	boundedContext bool,
) error {
	if session == nil || session.TrunkCache == nil {
		return fmt.Errorf("inference: %s session is invalid", label)
	}
	if err := r.validateCache(session.TrunkCache); err != nil {
		return fmt.Errorf("inference: %s trunk cache: %w", label, err)
	}
	if session.PendingHidden.Shape != tensor.MustShape(uint64(r.spec.EmbeddingLength), 1) ||
		session.Position == math.MaxUint32 {
		return fmt.Errorf("inference: %s session state is incompatible", label)
	}
	tokens := uint64(session.Position - session.MTPStart)
	validCache := session.Position >= effectiveCachePosition(session.TrunkCache) &&
		session.Position >= session.MTPStart &&
		session.Layer.Key.Shape.Rank == session.Layer.Value.Shape.Rank &&
		((session.Layer.Key.Shape.Rank == 0 && tokens == 0) ||
			(session.Layer.Key.Shape == tensor.MustShape(uint64(r.spec.KeyLength), uint64(r.spec.HeadCountKV), tokens) &&
				session.Layer.Value.Shape == tensor.MustShape(uint64(r.spec.ValueLength), uint64(r.spec.HeadCountKV), tokens)))
	if boundedContext && session.Layer.Key.Shape.Rank != 0 && tokens >= uint64(r.spec.ContextLength) {
		validCache = false
	}
	if !validCache {
		return fmt.Errorf("inference: %s layer cache is incompatible", label)
	}
	return nil
}

func (r *Runner) validateSingleHeadMTPTarget(
	target *Runner,
	mtpOnly bool,
	targetMTPOnly bool,
	label string,
) error {
	if !mtpOnly || target == nil || targetMTPOnly ||
		r.spec.Architecture != target.spec.Architecture ||
		r.spec.EmbeddingLength != target.spec.EmbeddingLength ||
		r.spec.VocabularySize != target.spec.VocabularySize ||
		r.spec.HeadCount != target.spec.HeadCount || r.spec.HeadCountKV != target.spec.HeadCountKV ||
		r.spec.KeyLength != target.spec.KeyLength || r.spec.ValueLength != target.spec.ValueLength ||
		r.spec.RopeDimensionCount != target.spec.RopeDimensionCount ||
		r.spec.RopeSections != target.spec.RopeSections ||
		len(target.weights.Layers) != int(target.spec.BlockCount) ||
		r.vocab == nil || target.vocab == nil || !slices.Equal(r.vocab.Tokens, target.vocab.Tokens) {
		return errors.New("inference: " + label + " sidecar target is incompatible")
	}
	return nil
}

func (r *Runner) mtpLayerInputs(
	ctx context.Context,
	builder *tensor.Builder,
	hostFeeds map[*tensor.Tensor]reference.Value,
	layer model.LayerWeights,
	prefix string,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	if r.hasPreloadedWeights() {
		return r.layerDeviceInputs(builder, layer)
	}
	hostLayer, err := model.LoadHostLayer(ctx, r.file, layer)
	if err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	graph, feeds, err := hostLayer.GraphInputs(builder, prefix)
	if err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	for node, value := range feeds {
		hostFeeds[node] = value
	}
	return graph, map[*tensor.Tensor]driver.DevicePtr{}, nil
}
