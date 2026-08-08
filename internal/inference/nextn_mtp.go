package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// NextNMTPSession: trunk snapshot plus draft state.
type NextNMTPSession = Qwen35MTPSession

// NewNextNMTPSession: bundled dense-tail setup.
func (r *Runner) NewNextNMTPSession(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (*NextNMTPSession, error) {
	if r == nil || len(tokenIDs) == 0 {
		return nil, errors.New("inference: NextN MTP inputs are invalid")
	}
	if err := r.validateNextNMTP(); err != nil {
		return nil, err
	}
	hidden, cache, err := r.ForwardCached(ctx, tokenIDs, nil)
	if err != nil {
		return nil, err
	}
	targetModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	position := effectiveCachePosition(cache)
	session := &NextNMTPSession{
		TrunkCache: cache, PendingHidden: lastHiddenColumn(hidden), MTPStart: position,
		Position: position, targetModel: targetModel,
	}
	if cache.DSATopK != nil {
		if cache.DSATopK.Shape.Rank != 2 || cache.DSATopK.Shape.Dims[0] != uint64(r.spec.IndexerTopK) ||
			cache.DSATopK.Shape.Dims[1] == 0 {
			return nil, errors.New("inference: GLM-DSA trunk top-k handoff is incompatible")
		}
		width := int(cache.DSATopK.Shape.Dims[0])
		value := reference.Value{
			Shape: tensor.MustShape(uint64(width), 1),
			Data:  slices.Clone(cache.DSATopK.Data[len(cache.DSATopK.Data)-width:]),
		}
		session.Layer.Auxiliary = &value
	}
	if err := r.validateNextNMTPSession(session); err != nil {
		return nil, err
	}
	return session, nil
}

// AdvanceNextNMTP: one draft step.
func (r *Runner) AdvanceNextNMTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *NextNMTPSession,
) (reference.Value, *NextNMTPSession, error) {
	if r == nil || session == nil || session.TrunkCache == nil {
		return reference.Value{}, nil, errors.New("inference: NextN MTP session is invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: NextN MTP runner is unavailable")
	}
	if err := r.validateNextNMTP(); err != nil {
		return reference.Value{}, nil, err
	}
	if err := r.validateNextNMTPSession(session); err != nil {
		return reference.Value{}, nil, err
	}
	if tokenID < 0 || int(tokenID) >= r.vocab.Len() {
		return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	mtp := &r.weights.NextNMTP[0]
	embeddingInfo := r.weights.TokenEmbedding
	if mtp.TokenEmbedding != nil {
		embeddingInfo = *mtp.TokenEmbedding
	}
	tokenEmbedding, err := r.loadRows(ctx, embeddingInfo, []uint32{uint32(tokenID)})
	if err != nil {
		return reference.Value{}, nil, err
	}
	graph := r.newInferenceGraphRuntime(ctx)
	builder := graph.builder
	tokenInput := graph.input("nextn_mtp.token", tokenEmbedding)
	hiddenInput := graph.input("nextn_mtp.hidden", session.PendingHidden)
	graphWeights, err := graph.layer(mtp.Layer, fmt.Sprintf("blk.%d.", r.spec.BlockCount))
	if err != nil {
		return reference.Value{}, nil, err
	}
	embeddingNorm, err := graph.weight(mtp.EmbeddingNorm)
	if err != nil {
		return reference.Value{}, nil, err
	}
	hiddenNorm, err := graph.weight(mtp.HiddenNorm)
	if err != nil {
		return reference.Value{}, nil, err
	}
	projection, err := graph.weight(mtp.EHProjection)
	if err != nil {
		return reference.Value{}, nil, err
	}
	current, err := model.BuildNextNMTPInput(
		builder, tokenInput, hiddenInput, embeddingNorm, hiddenNorm, projection, r.spec, 0,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	var pastKey, pastValue, pastIndexerKey, previousTopK *tensor.Tensor
	if session.Layer.Key.Shape.Rank != 0 {
		pastKey = graph.input("nextn_mtp.past_key", session.Layer.Key)
		pastValue = graph.input("nextn_mtp.past_value", session.Layer.Value)
	}
	if state, ok := session.Layer.States[model.CacheStateIndexerKey]; ok {
		pastIndexerKey = graph.input("nextn_mtp.past_indexer_key", state.Value)
	}
	if session.Layer.Auxiliary != nil {
		previousTopK = graph.input("nextn_mtp.previous_top_k", *session.Layer.Auxiliary)
	}
	draftLayer, err := r.draftLayerPlan(0)
	if err != nil {
		return reference.Value{}, nil, err
	}
	block, err := model.BuildNextNMTPBlockCachedWithDSAPlan(
		builder, current, r.spec, graphWeights, []uint32{session.Position},
		pastKey, pastValue, pastIndexerKey, previousTopK, 0, draftLayer,
	)
	if err != nil {
		return reference.Value{}, nil, err
	}
	outputNormInfo := r.outputNormTensorFor(mtp.OutputNorm, mtp.LayerOutputNorm)
	outputNorm, err := graph.weight(outputNormInfo)
	if err != nil {
		return reference.Value{}, nil, err
	}
	outputInfo := r.outputTensorFor(mtp.Output)
	output, err := graph.weight(outputInfo)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, nextHidden, err := model.BuildNextNMTPOutputs(builder, block.Output, outputNorm, output, r.spec, 0)
	if err != nil {
		return reference.Value{}, nil, err
	}
	outputs := []*tensor.Tensor{logits, nextHidden, block.Key, block.Value}
	indexerState, hasIndexerState := block.States[model.CacheStateIndexerKey]
	if hasIndexerState {
		outputs = append(outputs, indexerState.Value)
	}
	results, err := graph.execute(outputs...)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logitValue := results[logits]
	logitValue.Data = r.finalizeLogits(logitValue.Data)
	nextLayer := LayerCache{Key: results[block.Key], Value: results[block.Value]}
	if hasIndexerState {
		nextLayer.States = LayerStates{
			model.CacheStateIndexerKey: {
				Mode: indexerState.Mode, Value: results[indexerState.Value],
			},
		}
	}
	if block.Auxiliary != nil {
		value := *session.Layer.Auxiliary
		nextLayer.Auxiliary = &value
	}
	next := &NextNMTPSession{
		TrunkCache:    session.TrunkCache,
		Layer:         nextLayer,
		PendingHidden: results[nextHidden], MTPStart: session.MTPStart,
		Position: session.Position + 1, targetModel: session.targetModel,
	}
	return logitValue, next, nil
}

func (r *Runner) validateNextNMTP() error {
	if r == nil || !r.hasDraftSession(model.DraftNextNMTP, len(r.weights.NextNMTP)) {
		return errors.New("inference: model has no supported NextN MTP block")
	}
	return nil
}

func (r *Runner) validateNextNMTPSession(session *NextNMTPSession) error {
	if err := r.validateSingleHeadMTPSession(session, "NextN MTP", false); err != nil {
		return err
	}
	indexerState, hasIndexerState := session.Layer.States[model.CacheStateIndexerKey]
	profile := r.profile()
	if profile.Attention == model.AttentionDSA && profile.Auxiliary != model.AuxiliaryDSATopK {
		wantTokens := uint64(session.Position - session.MTPStart)
		if (wantTokens == 0 && hasIndexerState) || (wantTokens > 0 &&
			(!hasIndexerState || !indexerState.Mode.TokenAligned() ||
				indexerState.Value.Shape != tensor.MustShape(uint64(r.spec.IndexerKeyLength), 1, wantTokens))) {
			return errors.New("inference: NextN MTP indexer cache is incompatible")
		}
	} else if hasIndexerState {
		return errors.New("inference: NextN MTP session has unexpected indexer state")
	}
	if profile.Auxiliary == model.AuxiliaryDSATopK {
		if session.Layer.Auxiliary == nil ||
			session.Layer.Auxiliary.Shape != tensor.MustShape(uint64(r.spec.IndexerTopK), 1) {
			return errors.New("inference: GLM-DSA NextN MTP top-k handoff is incompatible")
		}
	} else if session.Layer.Auxiliary != nil {
		return errors.New("inference: NextN MTP session has unexpected auxiliary state")
	}
	return nil
}
