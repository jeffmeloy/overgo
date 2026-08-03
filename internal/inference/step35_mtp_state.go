package inference

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/checked"
	"llamacpp2go/internal/statecodec"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

const (
	step35MTPStateMagic  = "L2GS3501"
	step35MTPStateHeader = 104
)

// SaveStep35MTPSession: model-bound multi-head state.
func (r *Runner) SaveStep35MTPSession(session *Step35MTPSession) ([]byte, error) {
	if err := r.validateStep35MTP(); err != nil {
		return nil, err
	}
	return r.saveMultiHeadMTPSession(session)
}

// SaveHYV3MTPSession: model-bound HY-V3 head state.
func (r *Runner) SaveHYV3MTPSession(session *HYV3MTPSession) ([]byte, error) {
	if err := r.validateHYV3MTP(); err != nil {
		return nil, err
	}
	return r.saveMultiHeadMTPSession(session)
}

func (r *Runner) saveMultiHeadMTPSession(session *Step35MTPSession) ([]byte, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateMultiHeadMTP(); err != nil {
		return nil, err
	}
	if err := r.validateStep35MTPSession(session); err != nil {
		return nil, err
	}
	if session.targetModel == [32]byte{} {
		return nil, errors.New("inference: Step3.5 MTP target model binding is missing")
	}
	trunkData, err := r.SaveCache(session.TrunkCache)
	if err != nil {
		return nil, err
	}
	headData, err := marshalCache(&KVCache{
		Layers: session.Heads, Tokens: session.Position, Position: session.Position,
	})
	if err != nil {
		return nil, fmt.Errorf("inference: save Step3.5 MTP head caches: %w", err)
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if session.targetModel != draftModel {
		return nil, errors.New("inference: Step3.5 MTP session belongs to a different target model")
	}
	width := uint64(r.spec.EmbeddingLength)
	hiddenCount, ok := checked.Add64(1, uint64(len(session.DraftHidden)))
	hiddenElements, okElements := checked.Mul64(hiddenCount, width)
	hiddenBytes, okBytes := checked.Mul64(hiddenElements, 4)
	tokenBytes, okTokens := checked.Mul64(uint64(len(session.DraftTokens)), 4)
	total, okTotal := checked.Add64(
		uint64(step35MTPStateHeader), hiddenBytes, tokenBytes,
		uint64(len(trunkData)), uint64(len(headData)),
	)
	if !ok || !okElements || !okBytes || !okTokens || !okTotal || total > uint64(math.MaxInt) {
		return nil, errors.New("inference: Step3.5 MTP state exceeds addressable memory")
	}
	encoder := statecodec.NewEncoderCapacity(uint64(math.MaxInt), total)
	encoder.Raw([]byte(step35MTPStateMagic))
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

// LoadStep35MTPSession: bounded multi-head restore.
func (r *Runner) LoadStep35MTPSession(data []byte) (*Step35MTPSession, error) {
	if err := r.validateStep35MTP(); err != nil {
		return nil, err
	}
	return r.loadMultiHeadMTPSession(data)
}

// LoadHYV3MTPSession: bounded HY-V3 head restore.
func (r *Runner) LoadHYV3MTPSession(data []byte) (*HYV3MTPSession, error) {
	if err := r.validateHYV3MTP(); err != nil {
		return nil, err
	}
	return r.loadMultiHeadMTPSession(data)
}

func (r *Runner) loadMultiHeadMTPSession(data []byte) (*Step35MTPSession, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateMultiHeadMTP(); err != nil {
		return nil, err
	}
	decoder := statecodec.NewDecoder(data, uint64(math.MaxInt))
	if string(decoder.Raw(8)) != step35MTPStateMagic {
		return nil, errors.New("inference: Step3.5 MTP state has invalid magic or version")
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if string(decoder.Raw(32)) != string(draftModel[:]) {
		return nil, errors.New("inference: Step3.5 MTP state belongs to a different model")
	}
	var targetModel [32]byte
	copy(targetModel[:], decoder.Raw(32))
	if targetModel == [32]byte{} {
		return nil, errors.New("inference: Step3.5 MTP state target binding is invalid")
	}
	if targetModel != draftModel {
		return nil, errors.New("inference: Step3.5 MTP state belongs to a different target model")
	}
	mtpStart := decoder.U32()
	position := decoder.U32()
	headCount := decoder.U32()
	draftCount := decoder.U32()
	trunkLength := decoder.U64()
	headLength := decoder.U64()
	if decoder.Err() != nil {
		return nil, errors.New("inference: Step3.5 MTP state is truncated")
	}
	if headCount != uint32(len(r.multiHeadMTPWeights())) || draftCount > headCount ||
		position != mtpStart+draftCount || position == math.MaxUint32 || trunkLength == 0 || headLength == 0 {
		return nil, errors.New("inference: Step3.5 MTP state metadata is invalid")
	}
	width := uint64(r.spec.EmbeddingLength)
	hiddenCount, ok := checked.Add64(1, uint64(draftCount))
	hiddenElements, okElements := checked.Mul64(hiddenCount, width)
	hiddenBytes, okBytes := checked.Mul64(hiddenElements, 4)
	tokenBytes, okTokens := checked.Mul64(uint64(draftCount), 4)
	payload, okPayload := checked.Add64(hiddenBytes, tokenBytes, trunkLength, headLength)
	if !ok || !okElements || !okBytes || !okTokens || !okPayload || payload != decoder.Remaining() {
		return nil, errors.New("inference: Step3.5 MTP state payload lengths are invalid")
	}
	readHidden := func() reference.Value {
		hidden := reference.Value{
			Shape: tensor.MustShape(width, 1), Data: make([]float32, int(width)),
		}
		for index := range hidden.Data {
			hidden.Data[index] = decoder.F32()
		}
		return hidden
	}
	pendingHidden := readHidden()
	var draftTokens []tokenizer.TokenID
	if draftCount > 0 {
		draftTokens = make([]tokenizer.TokenID, int(draftCount))
	}
	for index := range draftTokens {
		draftTokens[index] = tokenizer.TokenID(decoder.I32())
	}
	var draftHidden []reference.Value
	if draftCount > 0 {
		draftHidden = make([]reference.Value, int(draftCount))
	}
	for index := range draftHidden {
		draftHidden[index] = readHidden()
	}
	trunk, err := r.LoadCache(decoder.Raw(trunkLength))
	if err != nil {
		return nil, fmt.Errorf("inference: load Step3.5 MTP trunk cache: %w", err)
	}
	headCache, err := unmarshalCache(decoder.Raw(headLength))
	if err != nil {
		return nil, fmt.Errorf("inference: load Step3.5 MTP head caches: %w", err)
	}
	if len(headCache.Layers) != int(headCount) || headCache.Tokens != position ||
		effectiveCachePosition(headCache) != position {
		return nil, errors.New("inference: Step3.5 MTP head cache metadata is invalid")
	}
	if err := decoder.Done(); err != nil {
		return nil, errors.New("inference: Step3.5 MTP state payload lengths are invalid")
	}
	session := &Step35MTPSession{
		TrunkCache: trunk, Heads: headCache.Layers, PendingHidden: pendingHidden,
		DraftTokens: draftTokens, DraftHidden: draftHidden,
		MTPStart: mtpStart, Position: position, targetModel: targetModel,
	}
	if err := r.validateStep35MTPSession(session); err != nil {
		return nil, err
	}
	return session, nil
}
