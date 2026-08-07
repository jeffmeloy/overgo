package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type singleHeadMTPBlock struct {
	output, key, value *tensor.Tensor
}

type singleHeadMTPAdapter struct {
	nodePrefix                         string
	layer                              model.LayerWeights
	embeddingNorm, hiddenNorm, project gguf.TensorInfo
	tokenEmbedding, outputNorm, output *gguf.TensorInfo
	buildInput                         func(*tensor.Builder, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor, model.Spec) (*tensor.Tensor, error)
	buildBlock                         func(*tensor.Builder, *tensor.Tensor, model.Spec, model.LayerGraphWeights, []uint32, *tensor.Tensor, *tensor.Tensor) (singleHeadMTPBlock, error)
	buildOutputs                       func(*tensor.Builder, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor, model.Spec) (*tensor.Tensor, *tensor.Tensor, error)
}

func (r *Runner) hasDraftSession(kind model.DraftKind, catalogs int) bool {
	if r == nil {
		return false
	}
	plan := r.profile().DraftPlan(r.spec.NextNPredictLayers)
	return plan.Kind == kind && plan.SessionEligible() && catalogs == int(plan.Heads)
}

func (r *Runner) advanceSingleHeadMTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *Qwen35MTPSession,
	adapter singleHeadMTPAdapter,
) (reference.Value, *Qwen35MTPSession, error) {
	embeddingInfo := r.weights.TokenEmbedding
	if adapter.tokenEmbedding != nil {
		embeddingInfo = *adapter.tokenEmbedding
	}
	tokenEmbedding, err := r.loadRows(ctx, embeddingInfo, []uint32{uint32(tokenID)})
	if err != nil {
		return reference.Value{}, nil, err
	}
	graph := r.newInferenceGraphRuntime(ctx)
	builder := graph.builder
	tokenInput := graph.input(adapter.nodePrefix+".token", tokenEmbedding)
	hiddenInput := graph.input(adapter.nodePrefix+".hidden", session.PendingHidden)
	graphWeights, err := graph.layer(adapter.layer, fmt.Sprintf("blk.%d.", r.spec.BlockCount))
	if err != nil {
		return reference.Value{}, nil, err
	}
	embeddingNorm, err := graph.weight(adapter.embeddingNorm)
	if err != nil {
		return reference.Value{}, nil, err
	}
	hiddenNorm, err := graph.weight(adapter.hiddenNorm)
	if err != nil {
		return reference.Value{}, nil, err
	}
	projection, err := graph.weight(adapter.project)
	if err != nil {
		return reference.Value{}, nil, err
	}
	current, err := adapter.buildInput(
		builder, tokenInput, hiddenInput, embeddingNorm, hiddenNorm, projection, r.spec,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	var pastKey, pastValue *tensor.Tensor
	if session.Layer.Key.Shape.Rank != 0 {
		pastKey = graph.input(adapter.nodePrefix+".past_key", session.Layer.Key)
		pastValue = graph.input(adapter.nodePrefix+".past_value", session.Layer.Value)
	}
	block, err := adapter.buildBlock(
		builder, current, r.spec, graphWeights, []uint32{session.Position}, pastKey, pastValue,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	outputNormInfo := r.outputNormTensorFor(adapter.outputNorm)
	outputNorm, err := graph.weight(outputNormInfo)
	if err != nil {
		return reference.Value{}, nil, err
	}
	outputInfo := r.outputTensorFor(adapter.output)
	output, err := graph.weight(outputInfo)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, nextHidden, err := adapter.buildOutputs(builder, block.output, outputNorm, output, r.spec)
	if err != nil {
		return reference.Value{}, nil, err
	}
	results, err := graph.execute(logits, nextHidden, block.key, block.value)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logitValue := results[logits]
	logitValue.Data = r.finalizeLogits(logitValue.Data)
	return logitValue, &Qwen35MTPSession{
		TrunkCache:    session.TrunkCache,
		Layer:         LayerCache{Key: results[block.key], Value: results[block.value]},
		PendingHidden: results[nextHidden], MTPStart: session.MTPStart,
		Position: session.Position + 1, targetModel: session.targetModel,
	}, nil
}

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
		Data:  slices.Clone(hidden.Data[len(hidden.Data)-width:]),
	}
}

func advanceTargetVerification[S any](
	ctx context.Context,
	target *Runner,
	token tokenizer.TokenID,
	cache *KVCache,
	advance func() (S, error),
	sync func(S, reference.Value, *KVCache),
) (reference.Value, S, error) {
	var zero S
	hidden, nextCache, err := target.ForwardCached(
		ctx, []tokenizer.TokenID{token}, cache,
	)
	if err != nil {
		return reference.Value{}, zero, err
	}
	logits, err := target.projectHiddenLogits(ctx, hidden)
	if err != nil {
		return reference.Value{}, zero, err
	}
	next, err := advance()
	if err != nil {
		return reference.Value{}, zero, err
	}
	sync(next, hidden, nextCache)
	return logits, next, nil
}

func (r *Runner) advanceSingleHeadMTPVerification(
	ctx context.Context,
	target *Runner,
	token tokenizer.TokenID,
	session *Qwen35MTPSession,
	targetCache *KVCache,
	advance func(tokenizer.TokenID, *Qwen35MTPSession) (reference.Value, *Qwen35MTPSession, error),
) (reference.Value, *Qwen35MTPSession, error) {
	return advanceTargetVerification(
		ctx, target, token, targetCache,
		func() (*Qwen35MTPSession, error) {
			_, next, err := advance(token, session)
			return next, err
		},
		func(next *Qwen35MTPSession, hidden reference.Value, cache *KVCache) {
			next.PendingHidden = lastHiddenColumn(hidden)
			next.TrunkCache = cache
		},
	)
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
		r.spec.Profile() != target.spec.Profile() ||
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

func (r *Runner) validateSingleHeadMTPVerificationTarget(
	target *Runner,
	session *Qwen35MTPSession,
	mtpOnly bool,
	label string,
	validate func(*Runner) error,
) error {
	if mtpOnly {
		if err := validate(target); err != nil {
			return err
		}
	} else if r != target {
		return fmt.Errorf(
			"inference: bundled %s verification requires its owning target runner", label,
		)
	}
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return err
	}
	if session.targetModel != targetModel {
		return fmt.Errorf("inference: %s session belongs to a different target model", label)
	}
	return nil
}
