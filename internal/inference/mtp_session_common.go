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

type singleHeadMTPAdapter struct {
	nodePrefix                         string
	layer                              model.LayerWeights
	program                            model.CompiledLayerProgram
	embeddingNorm, hiddenNorm, project gguf.TensorInfo
	tokenEmbedding, outputNorm, output *gguf.TensorInfo
}

func (r *Runner) hasDraftSession(kind model.DraftKind, catalogs int) bool {
	if r == nil {
		return false
	}
	plan := r.program.Model.Draft()
	return plan.Kind == kind && plan.SessionEligible() && catalogs == int(plan.Heads)
}

func (r *Runner) advanceSingleHeadMTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *MTPSession,
	adapter singleHeadMTPAdapter,
) (reference.Value, *MTPSession, error) {
	embeddingInfo := r.weights.TokenEmbedding
	if adapter.tokenEmbedding != nil {
		embeddingInfo = *adapter.tokenEmbedding
	}
	tokenEmbedding, err := r.gatherTensor(ctx, embeddingInfo, []uint32{uint32(tokenID)})
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
	current, err := adapter.program.BuildDraftInput(
		builder, tokenInput, hiddenInput, embeddingNorm, hiddenNorm, projection,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	var pastKey, pastValue *tensor.Tensor
	if session.Layer.Key.Defined() {
		pastKey = graph.input(adapter.nodePrefix+".past_key", session.Layer.Key)
		pastValue = graph.input(adapter.nodePrefix+".past_value", session.Layer.Value)
	}
	plan := adapter.program.Layer()
	block, err := adapter.program.Build(model.CachedBlockContext{
		Builder: builder, Input: current, Positions: []uint32{session.Position},
		PastKey: pastKey, PastValue: pastValue, Layer: plan.Layer,
		CacheWrite: tensor.CacheWriteConcat, Sequences: 1,
	}, graphWeights)
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
	logits, nextHidden, err := adapter.program.BuildDraftOutputs(builder, block.Output, outputNorm, output)
	if err != nil {
		return reference.Value{}, nil, err
	}
	results, err := graph.execute(logits, nextHidden, block.Key, block.Value)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logitValue := results[logits]
	logitValue.Data = r.finalizeLogits(logitValue.Data)
	return logitValue, &MTPSession{
		TrunkCache:    session.TrunkCache,
		Layer:         LayerCache{Key: results[block.Key], Value: results[block.Value]},
		PendingHidden: results[nextHidden], MTPStart: session.MTPStart,
		Position: session.Position + 1, targetModel: session.targetModel,
	}, nil
}

func (target *Runner) newSingleHeadMTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*MTPSession, error) {
	hidden, cache, err := target.ForwardCached(ctx, tokenIDs, nil)
	if err != nil {
		return nil, err
	}
	last := hidden.LastRowView().Clone()
	targetModel, err := target.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	position := effectiveCachePosition(cache)
	return &MTPSession{
		TrunkCache: cache, PendingHidden: last, MTPStart: position, Position: position,
		targetModel: targetModel,
	}, nil
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
	session *MTPSession,
	targetCache *KVCache,
	advance func(tokenizer.TokenID, *MTPSession) (reference.Value, *MTPSession, error),
) (reference.Value, *MTPSession, error) {
	return advanceTargetVerification(
		ctx, target, token, targetCache,
		func() (*MTPSession, error) {
			_, next, err := advance(token, session)
			return next, err
		},
		func(next *MTPSession, hidden reference.Value, cache *KVCache) {
			next.PendingHidden = hidden.LastRowView().Clone()
			next.TrunkCache = cache
		},
	)
}

func (r *Runner) validateSingleHeadMTPSession(
	session *MTPSession,
	label string,
	boundedContext bool,
) error {
	if session == nil || session.TrunkCache == nil {
		return fmt.Errorf("inference: %s session is invalid", label)
	}
	if err := r.validateCache(session.TrunkCache); err != nil {
		return fmt.Errorf("inference: %s trunk cache: %w", label, err)
	}
	if r.spec.ValidateSequenceRow(session.PendingHidden) != nil || session.Position == math.MaxUint32 {
		return fmt.Errorf("inference: %s session state is incompatible", label)
	}
	tokens := uint64(session.Position - session.MTPStart)
	emptyCache := !session.Layer.Key.Defined() && !session.Layer.Value.Defined()
	validCache := session.Position >= effectiveCachePosition(session.TrunkCache) &&
		session.Position >= session.MTPStart &&
		((emptyCache && tokens == 0) ||
			(tensor.HasDimensions(session.Layer.Key.Shape, uint64(r.spec.KeyLength), uint64(r.spec.HeadCountKV), tokens) &&
				tensor.HasDimensions(session.Layer.Value.Shape, uint64(r.spec.ValueLength), uint64(r.spec.HeadCountKV), tokens))) &&
		(!boundedContext || !session.Layer.Key.Defined() || tokens < uint64(r.spec.ContextLength))
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
		!r.program.Model.SameExecutionProfile(target.program.Model) ||
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
	session *MTPSession,
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
