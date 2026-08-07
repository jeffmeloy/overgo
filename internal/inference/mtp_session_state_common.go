package inference

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/statecodec"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
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
	encoder := statecodec.NewEncoder(uint64(math.MaxInt))
	encoder.Raw([]byte(codec.magic))
	encoder.Raw(draftModel[:])
	encoder.Raw(session.targetModel[:])
	encoder.U32(session.MTPStart)
	encoder.U32(session.Position)
	encoder.U64(uint64(len(trunkData)))
	encoder.U64(uint64(len(layerData)))
	for _, value := range session.PendingHidden.Data {
		encoder.F32(value)
	}
	encoder.Raw(trunkData)
	encoder.Raw(layerData)
	output, err := encoder.Data()
	if err != nil {
		return nil, fmt.Errorf("inference: %s state exceeds addressable memory", codec.label)
	}
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
	decoder := statecodec.NewDecoder(data, uint64(math.MaxInt))
	magic := decoder.Raw(8)
	draftSignature := decoder.Raw(32)
	targetSignature := decoder.Raw(32)
	mtpStart := decoder.U32()
	position := decoder.U32()
	trunkLength := decoder.U64()
	layerLength := decoder.U64()
	if decoder.Err() != nil {
		return nil, fmt.Errorf("inference: %s state is truncated", codec.label)
	}
	if string(magic) != codec.magic {
		return nil, fmt.Errorf("inference: %s state has invalid magic or version", codec.label)
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if string(draftSignature) != string(draftModel[:]) {
		return nil, fmt.Errorf("inference: %s state belongs to a different draft model", codec.label)
	}
	var targetModel [32]byte
	copy(targetModel[:], targetSignature)
	if targetModel == [32]byte{} {
		return nil, fmt.Errorf("inference: %s state target binding is invalid", codec.label)
	}
	if position < mtpStart || position == math.MaxUint32 ||
		(position == mtpStart) != (layerLength == 0) {
		return nil, fmt.Errorf("inference: %s state positions are invalid", codec.label)
	}
	hiddenBytes, ok := checked.Bytes(uint64(r.spec.EmbeddingLength), 4)
	if !ok {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", codec.label)
	}
	payload := decoder.Remaining()
	if hiddenBytes > payload || trunkLength > payload-hiddenBytes ||
		layerLength != payload-hiddenBytes-trunkLength || trunkLength == 0 {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", codec.label)
	}
	hidden := reference.Value{
		Shape: tensor.MustShape(uint64(r.spec.EmbeddingLength), 1),
		Data:  make([]float32, int(r.spec.EmbeddingLength)),
	}
	for index := range hidden.Data {
		hidden.Data[index] = decoder.F32()
	}
	trunkData := decoder.Raw(trunkLength)
	layerData := decoder.Raw(layerLength)
	if decoder.Done() != nil {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", codec.label)
	}
	trunk, err := r.LoadCache(trunkData)
	if err != nil {
		return nil, fmt.Errorf("inference: load %s trunk cache: %w", codec.label, err)
	}
	session := &Qwen35MTPSession{
		TrunkCache: trunk, PendingHidden: hidden, MTPStart: mtpStart,
		Position: position, targetModel: targetModel,
	}
	if layerLength > 0 {
		layerCache, err := unmarshalCache(layerData)
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
