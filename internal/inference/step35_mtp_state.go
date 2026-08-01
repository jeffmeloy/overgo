package inference

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

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
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateStep35MTP(); err != nil {
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
	hiddenBytes := (uint64(1) + uint64(len(session.DraftHidden))) * width * 4
	tokenBytes := uint64(len(session.DraftTokens)) * 4
	total := uint64(step35MTPStateHeader) + hiddenBytes + tokenBytes +
		uint64(len(trunkData)) + uint64(len(headData))
	if total > uint64(maxIntValue()) {
		return nil, errors.New("inference: Step3.5 MTP state exceeds addressable memory")
	}
	output := make([]byte, int(total))
	copy(output, step35MTPStateMagic)
	copy(output[8:40], draftModel[:])
	copy(output[40:72], session.targetModel[:])
	binary.LittleEndian.PutUint32(output[72:], session.MTPStart)
	binary.LittleEndian.PutUint32(output[76:], session.Position)
	binary.LittleEndian.PutUint32(output[80:], uint32(len(session.Heads)))
	binary.LittleEndian.PutUint32(output[84:], uint32(len(session.DraftTokens)))
	binary.LittleEndian.PutUint64(output[88:], uint64(len(trunkData)))
	binary.LittleEndian.PutUint64(output[96:], uint64(len(headData)))
	offset := step35MTPStateHeader
	writeHidden := func(hidden reference.Value) {
		for _, value := range hidden.Data {
			binary.LittleEndian.PutUint32(output[offset:], math.Float32bits(value))
			offset += 4
		}
	}
	writeHidden(session.PendingHidden)
	for _, token := range session.DraftTokens {
		binary.LittleEndian.PutUint32(output[offset:], uint32(int32(token)))
		offset += 4
	}
	for _, hidden := range session.DraftHidden {
		writeHidden(hidden)
	}
	copy(output[offset:], trunkData)
	offset += len(trunkData)
	copy(output[offset:], headData)
	return output, nil
}

// LoadStep35MTPSession: bounded multi-head restore.
func (r *Runner) LoadStep35MTPSession(data []byte) (*Step35MTPSession, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateStep35MTP(); err != nil {
		return nil, err
	}
	if len(data) < step35MTPStateHeader {
		return nil, errors.New("inference: Step3.5 MTP state is truncated")
	}
	if string(data[:8]) != step35MTPStateMagic {
		return nil, errors.New("inference: Step3.5 MTP state has invalid magic or version")
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if string(data[8:40]) != string(draftModel[:]) {
		return nil, errors.New("inference: Step3.5 MTP state belongs to a different model")
	}
	var targetModel [32]byte
	copy(targetModel[:], data[40:72])
	if targetModel == [32]byte{} {
		return nil, errors.New("inference: Step3.5 MTP state target binding is invalid")
	}
	if targetModel != draftModel {
		return nil, errors.New("inference: Step3.5 MTP state belongs to a different target model")
	}
	mtpStart := binary.LittleEndian.Uint32(data[72:])
	position := binary.LittleEndian.Uint32(data[76:])
	headCount := binary.LittleEndian.Uint32(data[80:])
	draftCount := binary.LittleEndian.Uint32(data[84:])
	trunkLength := binary.LittleEndian.Uint64(data[88:])
	headLength := binary.LittleEndian.Uint64(data[96:])
	if headCount != uint32(len(r.weights.Step35MTP)) || draftCount > headCount ||
		position != mtpStart+draftCount || position == math.MaxUint32 || trunkLength == 0 || headLength == 0 {
		return nil, errors.New("inference: Step3.5 MTP state metadata is invalid")
	}
	width := uint64(r.spec.EmbeddingLength)
	hiddenBytes := (uint64(1) + uint64(draftCount)) * width * 4
	tokenBytes := uint64(draftCount) * 4
	payload := uint64(len(data) - step35MTPStateHeader)
	if hiddenBytes > payload || tokenBytes > payload-hiddenBytes ||
		trunkLength > payload-hiddenBytes-tokenBytes ||
		headLength != payload-hiddenBytes-tokenBytes-trunkLength {
		return nil, errors.New("inference: Step3.5 MTP state payload lengths are invalid")
	}
	offset := step35MTPStateHeader
	readHidden := func() reference.Value {
		hidden := reference.Value{
			Shape: tensor.MustShape(width, 1), Data: make([]float32, int(width)),
		}
		for index := range hidden.Data {
			hidden.Data[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
			offset += 4
		}
		return hidden
	}
	pendingHidden := readHidden()
	var draftTokens []tokenizer.TokenID
	if draftCount > 0 {
		draftTokens = make([]tokenizer.TokenID, int(draftCount))
	}
	for index := range draftTokens {
		draftTokens[index] = tokenizer.TokenID(int32(binary.LittleEndian.Uint32(data[offset:])))
		offset += 4
	}
	var draftHidden []reference.Value
	if draftCount > 0 {
		draftHidden = make([]reference.Value, int(draftCount))
	}
	for index := range draftHidden {
		draftHidden[index] = readHidden()
	}
	trunkEnd := offset + int(trunkLength)
	trunk, err := r.LoadCache(data[offset:trunkEnd])
	if err != nil {
		return nil, fmt.Errorf("inference: load Step3.5 MTP trunk cache: %w", err)
	}
	headCache, err := unmarshalCache(data[trunkEnd:])
	if err != nil {
		return nil, fmt.Errorf("inference: load Step3.5 MTP head caches: %w", err)
	}
	if len(headCache.Layers) != int(headCount) || headCache.Tokens != position ||
		effectiveCachePosition(headCache) != position {
		return nil, errors.New("inference: Step3.5 MTP head cache metadata is invalid")
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
