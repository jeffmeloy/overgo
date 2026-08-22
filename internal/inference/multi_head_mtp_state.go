package inference

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/statecodec"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

const multiHeadMTPStateMagic = "L2GMHM02"

// SaveMultiHeadMTPSession serializes model-bound multi-head state.
func (r *Runner) SaveMultiHeadMTPSession(session *MultiHeadMTPSession) ([]byte, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	if _, err := r.multiHeadMTP(); err != nil {
		return nil, err
	}
	if err := r.validateMultiHeadMTPSession(session); err != nil {
		return nil, err
	}
	if !checked.Nonzero(session.targetModel) {
		return nil, errors.New("inference: multi-head MTP target model binding is missing")
	}
	trunkData, err := r.SaveCache(session.TrunkCache)
	if err != nil {
		return nil, err
	}
	headData, err := marshalCache(&KVCache{
		Layers: session.Heads, Tokens: session.Position, Position: session.Position,
	})
	if err != nil {
		return nil, fmt.Errorf("inference: save multi-head MTP head caches: %w", err)
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if session.targetModel != draftModel {
		return nil, errors.New("inference: multi-head MTP session belongs to a different target model")
	}
	width := uint64(r.spec.EmbeddingLength)
	hiddenCount, ok := checked.Add64(tensor.SingletonExtent, uint64(len(session.DraftHidden)))
	hiddenElements, okElements := checked.Mul64(hiddenCount, width)
	hiddenBytes, okBytes := checked.Mul64(hiddenElements, binaryschema.Uint32Bytes)
	tokenBytes, okTokens := checked.Mul64(uint64(len(session.DraftTokens)), binaryschema.Uint32Bytes)
	u32Header, validU32Header := checked.Mul64(uint64(tensor.MaxDimensions), binaryschema.Uint32Bytes)
	u64Header, validU64Header := checked.Mul64(uint64(tensor.PairedExtent), binaryschema.Uint64Bytes)
	headerBytes, validHeader := checked.Add64(
		uint64(len(multiHeadMTPStateMagic)), uint64(len(draftModel)), uint64(len(session.targetModel)),
		u32Header, u64Header,
	)
	total, okTotal := checked.Add64(
		headerBytes, hiddenBytes, tokenBytes,
		uint64(len(trunkData)), uint64(len(headData)),
	)
	if !ok || !okElements || !okBytes || !okTokens || !validU32Header || !validU64Header ||
		!validHeader || !okTotal || total > uint64(math.MaxInt) {
		return nil, errors.New("inference: multi-head MTP state exceeds addressable memory")
	}
	encoder := statecodec.NewEncoderCapacity(uint64(math.MaxInt), total)
	encoder.Raw([]byte(multiHeadMTPStateMagic))
	encoder.Raw(draftModel[:])
	encoder.Raw(session.targetModel[:])
	encoder.U32(session.MTPStart)
	encoder.U32(session.Position)
	encoder.U32(uint32(len(session.Heads)))
	encoder.U32(uint32(len(session.DraftTokens)))
	encoder.U64(uint64(len(trunkData)))
	encoder.U64(uint64(len(headData)))
	writeHidden := func(hidden reference.Value) {
		for _, value := range hidden.Data {
			encoder.F32(value)
		}
	}
	writeHidden(session.PendingHidden)
	for _, token := range session.DraftTokens {
		encoder.I32(int32(token))
	}
	for _, hidden := range session.DraftHidden {
		writeHidden(hidden)
	}
	encoder.Raw(trunkData)
	encoder.Raw(headData)
	return encoder.Data()
}

// LoadMultiHeadMTPSession restores bounded multi-head state.
func (r *Runner) LoadMultiHeadMTPSession(data []byte) (*MultiHeadMTPSession, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	plan, err := r.multiHeadMTP()
	if err != nil {
		return nil, err
	}
	decoder, err := r.modelStateDecoder(data, multiHeadMTPStateMagic, "multi-head MTP")
	if err != nil {
		return nil, err
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	var targetModel [sha256.Size]byte
	copy(targetModel[:], decoder.Raw(sha256.Size))
	if !checked.Nonzero(targetModel) {
		return nil, errors.New("inference: multi-head MTP state target binding is invalid")
	}
	if targetModel != draftModel {
		return nil, errors.New("inference: multi-head MTP state belongs to a different target model")
	}
	mtpStart := decoder.U32()
	position := decoder.U32()
	headCount := decoder.U32()
	draftCount := decoder.U32()
	trunkLength := decoder.U64()
	headLength := decoder.U64()
	if decoder.Err() != nil {
		return nil, errors.New("inference: multi-head MTP state is truncated")
	}
	if headCount != plan.Heads || draftCount > headCount ||
		position != mtpStart+draftCount || position == math.MaxUint32 ||
		!checked.Nonzero(trunkLength) || !checked.Nonzero(headLength) {
		return nil, errors.New("inference: multi-head MTP state metadata is invalid")
	}
	width := uint64(r.spec.EmbeddingLength)
	hiddenCount, ok := checked.Add64(tensor.SingletonExtent, uint64(draftCount))
	hiddenElements, okElements := checked.Mul64(hiddenCount, width)
	hiddenBytes, okBytes := checked.Mul64(hiddenElements, binaryschema.Uint32Bytes)
	tokenBytes, okTokens := checked.Mul64(uint64(draftCount), binaryschema.Uint32Bytes)
	payload, okPayload := checked.Add64(hiddenBytes, tokenBytes, trunkLength, headLength)
	if !ok || !okElements || !okBytes || !okTokens || !okPayload || payload != decoder.Remaining() {
		return nil, errors.New("inference: multi-head MTP state payload lengths are invalid")
	}
	pendingHidden := decodeHiddenState(decoder, width)
	var draftTokens []tokenizer.TokenID
	if checked.Nonzero(draftCount) {
		draftTokens = make([]tokenizer.TokenID, int(draftCount))
	}
	for index := range draftTokens {
		draftTokens[index] = tokenizer.TokenID(decoder.I32())
	}
	var draftHidden []reference.Value
	if checked.Nonzero(draftCount) {
		draftHidden = make([]reference.Value, int(draftCount))
	}
	for index := range draftHidden {
		draftHidden[index] = decodeHiddenState(decoder, width)
	}
	trunk, err := r.LoadCache(decoder.Raw(trunkLength))
	if err != nil {
		return nil, fmt.Errorf("inference: load multi-head MTP trunk cache: %w", err)
	}
	headCache, err := unmarshalCache(decoder.Raw(headLength))
	if err != nil {
		return nil, fmt.Errorf("inference: load multi-head MTP head caches: %w", err)
	}
	if len(headCache.Layers) != int(headCount) || headCache.Tokens != position ||
		effectiveCachePosition(headCache) != position {
		return nil, errors.New("inference: multi-head MTP head cache metadata is invalid")
	}
	if err := decoder.Done(); err != nil {
		return nil, errors.New("inference: multi-head MTP state payload lengths are invalid")
	}
	session := &MultiHeadMTPSession{
		TrunkCache: trunk, Heads: headCache.Layers, PendingHidden: pendingHidden,
		DraftTokens: draftTokens, DraftHidden: draftHidden,
		MTPStart: mtpStart, Position: position, targetModel: targetModel,
	}
	if err := r.validateMultiHeadMTPSession(session); err != nil {
		return nil, err
	}
	return session, nil
}
