package inference

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

const singleHeadMTPStateHeader = 96

type singleHeadMTPStateCodec struct {
	magic           string
	label           string
	validateModel   func() error
	validateSession func(*Qwen35MTPSession) error
}

func (r *Runner) saveSingleHeadMTPSession(
	session *Qwen35MTPSession,
	codec singleHeadMTPStateCodec,
) ([]byte, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := codec.validateModel(); err != nil {
		return nil, err
	}
	if err := codec.validateSession(session); err != nil {
		return nil, err
	}
	if session.targetModel == [32]byte{} {
		return nil, fmt.Errorf("inference: %s target model binding is missing", codec.label)
	}
	trunkData, err := r.SaveCache(session.TrunkCache)
	if err != nil {
		return nil, err
	}
	var layerData []byte
	draftTokens := session.Position - session.MTPStart
	if draftTokens > 0 {
		layerData, err = marshalCache(&KVCache{
			Layers: []LayerCache{session.Layer}, Tokens: draftTokens, Position: session.Position,
		})
		if err != nil {
			return nil, fmt.Errorf("inference: save %s layer cache: %w", codec.label, err)
		}
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	hiddenBytes := uint64(r.spec.EmbeddingLength) * 4
	total := uint64(singleHeadMTPStateHeader) + hiddenBytes +
		uint64(len(trunkData)) + uint64(len(layerData))
	if total > uint64(maxIntValue()) {
		return nil, fmt.Errorf("inference: %s state exceeds addressable memory", codec.label)
	}
	output := make([]byte, int(total))
	copy(output, codec.magic)
	copy(output[8:40], draftModel[:])
	copy(output[40:72], session.targetModel[:])
	binary.LittleEndian.PutUint32(output[72:], session.MTPStart)
	binary.LittleEndian.PutUint32(output[76:], session.Position)
	binary.LittleEndian.PutUint64(output[80:], uint64(len(trunkData)))
	binary.LittleEndian.PutUint64(output[88:], uint64(len(layerData)))
	offset := singleHeadMTPStateHeader
	for _, value := range session.PendingHidden.Data {
		binary.LittleEndian.PutUint32(output[offset:], math.Float32bits(value))
		offset += 4
	}
	copy(output[offset:], trunkData)
	offset += len(trunkData)
	copy(output[offset:], layerData)
	return output, nil
}

func (r *Runner) loadSingleHeadMTPSession(
	data []byte,
	codec singleHeadMTPStateCodec,
) (*Qwen35MTPSession, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := codec.validateModel(); err != nil {
		return nil, err
	}
	if len(data) < singleHeadMTPStateHeader {
		return nil, fmt.Errorf("inference: %s state is truncated", codec.label)
	}
	if string(data[:8]) != codec.magic {
		return nil, fmt.Errorf("inference: %s state has invalid magic or version", codec.label)
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if string(data[8:40]) != string(draftModel[:]) {
		return nil, fmt.Errorf("inference: %s state belongs to a different draft model", codec.label)
	}
	var targetModel [32]byte
	copy(targetModel[:], data[40:72])
	if targetModel == [32]byte{} {
		return nil, fmt.Errorf("inference: %s state target binding is invalid", codec.label)
	}
	mtpStart := binary.LittleEndian.Uint32(data[72:])
	position := binary.LittleEndian.Uint32(data[76:])
	trunkLength := binary.LittleEndian.Uint64(data[80:])
	layerLength := binary.LittleEndian.Uint64(data[88:])
	if position < mtpStart || position == math.MaxUint32 ||
		(position == mtpStart) != (layerLength == 0) {
		return nil, fmt.Errorf("inference: %s state positions are invalid", codec.label)
	}
	hiddenBytes := uint64(r.spec.EmbeddingLength) * 4
	payload := uint64(len(data) - singleHeadMTPStateHeader)
	if hiddenBytes > payload || trunkLength > payload-hiddenBytes ||
		layerLength != payload-hiddenBytes-trunkLength || trunkLength == 0 {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", codec.label)
	}
	hidden := reference.Value{
		Shape: tensor.MustShape(uint64(r.spec.EmbeddingLength), 1),
		Data:  make([]float32, int(r.spec.EmbeddingLength)),
	}
	offset := singleHeadMTPStateHeader
	for index := range hidden.Data {
		hidden.Data[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}
	trunkEnd := offset + int(trunkLength)
	trunk, err := r.LoadCache(data[offset:trunkEnd])
	if err != nil {
		return nil, fmt.Errorf("inference: load %s trunk cache: %w", codec.label, err)
	}
	session := &Qwen35MTPSession{
		TrunkCache: trunk, PendingHidden: hidden, MTPStart: mtpStart,
		Position: position, targetModel: targetModel,
	}
	if layerLength > 0 {
		layerCache, err := unmarshalCache(data[trunkEnd:])
		if err != nil {
			return nil, fmt.Errorf("inference: load %s layer cache: %w", codec.label, err)
		}
		if len(layerCache.Layers) != 1 || layerCache.Tokens != position-mtpStart ||
			layerCache.Position != position {
			return nil, fmt.Errorf("inference: %s layer cache metadata is invalid", codec.label)
		}
		session.Layer = layerCache.Layers[0]
	}
	if err := codec.validateSession(session); err != nil {
		return nil, err
	}
	return session, nil
}
