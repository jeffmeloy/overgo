package inference

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

const (
	qwen35MTPStateMagic  = "L2GMTP01"
	qwen35MTPStateHeader = 96
)

// SaveQwen35MTPSession: draft/target-bound resumable state.
func (r *Runner) SaveQwen35MTPSession(session *Qwen35MTPSession) ([]byte, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateQwen35MTP(); err != nil {
		return nil, err
	}
	if err := r.validateQwen35MTPSession(session); err != nil {
		return nil, err
	}
	if session.targetModel == [32]byte{} {
		return nil, errors.New("inference: Qwen3.5 MTP target model binding is missing")
	}
	trunkData, err := r.SaveCache(session.TrunkCache)
	if err != nil {
		return nil, err
	}
	var mtpData []byte
	draftTokens := session.Position - session.MTPStart
	if draftTokens > 0 {
		mtpData, err = marshalCache(&KVCache{
			Layers: []LayerCache{session.Layer}, Tokens: draftTokens, Position: session.Position,
		})
		if err != nil {
			return nil, fmt.Errorf("inference: save Qwen3.5 MTP layer cache: %w", err)
		}
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	hiddenBytes := uint64(r.spec.EmbeddingLength) * 4
	total := uint64(qwen35MTPStateHeader) + hiddenBytes + uint64(len(trunkData)) + uint64(len(mtpData))
	if total > uint64(maxIntValue()) {
		return nil, errors.New("inference: Qwen3.5 MTP state exceeds addressable memory")
	}
	output := make([]byte, int(total))
	copy(output, qwen35MTPStateMagic)
	copy(output[8:40], draftModel[:])
	copy(output[40:72], session.targetModel[:])
	binary.LittleEndian.PutUint32(output[72:], session.MTPStart)
	binary.LittleEndian.PutUint32(output[76:], session.Position)
	binary.LittleEndian.PutUint64(output[80:], uint64(len(trunkData)))
	binary.LittleEndian.PutUint64(output[88:], uint64(len(mtpData)))
	offset := qwen35MTPStateHeader
	for _, value := range session.PendingHidden.Data {
		binary.LittleEndian.PutUint32(output[offset:], math.Float32bits(value))
		offset += 4
	}
	copy(output[offset:], trunkData)
	offset += len(trunkData)
	copy(output[offset:], mtpData)
	return output, nil
}

// LoadQwen35MTPSession: bounded model-bound restore.
func (r *Runner) LoadQwen35MTPSession(data []byte) (*Qwen35MTPSession, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateQwen35MTP(); err != nil {
		return nil, err
	}
	if len(data) < qwen35MTPStateHeader {
		return nil, errors.New("inference: Qwen3.5 MTP state is truncated")
	}
	if string(data[:8]) != qwen35MTPStateMagic {
		return nil, errors.New("inference: Qwen3.5 MTP state has invalid magic or version")
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if string(data[8:40]) != string(draftModel[:]) {
		return nil, errors.New("inference: Qwen3.5 MTP state belongs to a different draft model")
	}
	var targetModel [32]byte
	copy(targetModel[:], data[40:72])
	if targetModel == [32]byte{} {
		return nil, errors.New("inference: Qwen3.5 MTP state target binding is invalid")
	}
	mtpStart := binary.LittleEndian.Uint32(data[72:])
	position := binary.LittleEndian.Uint32(data[76:])
	trunkLength := binary.LittleEndian.Uint64(data[80:])
	mtpLength := binary.LittleEndian.Uint64(data[88:])
	if position < mtpStart || position == math.MaxUint32 ||
		(position == mtpStart) != (mtpLength == 0) {
		return nil, errors.New("inference: Qwen3.5 MTP state positions are invalid")
	}
	hiddenBytes := uint64(r.spec.EmbeddingLength) * 4
	payload := uint64(len(data) - qwen35MTPStateHeader)
	if hiddenBytes > payload || trunkLength > payload-hiddenBytes ||
		mtpLength != payload-hiddenBytes-trunkLength || trunkLength == 0 {
		return nil, errors.New("inference: Qwen3.5 MTP state payload lengths are invalid")
	}
	hidden := reference.Value{
		Shape: tensor.MustShape(uint64(r.spec.EmbeddingLength), 1),
		Data:  make([]float32, int(r.spec.EmbeddingLength)),
	}
	offset := qwen35MTPStateHeader
	for index := range hidden.Data {
		hidden.Data[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}
	trunkEnd := offset + int(trunkLength)
	trunk, err := r.LoadCache(data[offset:trunkEnd])
	if err != nil {
		return nil, fmt.Errorf("inference: load Qwen3.5 MTP trunk cache: %w", err)
	}
	session := &Qwen35MTPSession{
		TrunkCache: trunk, PendingHidden: hidden,
		MTPStart: mtpStart, Position: position, targetModel: targetModel,
	}
	if mtpLength > 0 {
		mtpCache, cacheErr := unmarshalCache(data[trunkEnd:])
		if cacheErr != nil {
			return nil, fmt.Errorf("inference: load Qwen3.5 MTP layer cache: %w", cacheErr)
		}
		if len(mtpCache.Layers) != 1 || mtpCache.Tokens != position-mtpStart ||
			mtpCache.Position != position {
			return nil, errors.New("inference: Qwen3.5 MTP layer cache metadata is invalid")
		}
		session.Layer = mtpCache.Layers[0]
	}
	if err := r.validateQwen35MTPSession(session); err != nil {
		return nil, err
	}
	return session, nil
}
