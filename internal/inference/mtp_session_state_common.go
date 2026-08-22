package inference

import (
	"crypto/sha256"
	"fmt"
	"math"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/statecodec"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const (
	singleHeadMTPStateMagic = "L2GMTP02"
	mtpLabel                = "MTP"
)

// SaveMTPSession encodes compiled-program-bound single-head draft state.
func (r *Runner) SaveMTPSession(session *MTPSession) ([]byte, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	_, _, err := r.singleHeadMTP()
	if err != nil {
		return nil, err
	}
	if err := r.validateMTPSession(session); err != nil {
		return nil, err
	}
	if !checked.Nonzero(session.targetModel) {
		return nil, fmt.Errorf("inference: %s target model binding is missing", mtpLabel)
	}
	trunkData, err := r.SaveCache(session.TrunkCache)
	if err != nil {
		return nil, err
	}
	var layerData []byte
	draftTokens := session.Position - session.MTPStart
	if checked.Nonzero(draftTokens) {
		layerData, err = marshalCache(&KVCache{
			Layers: []LayerCache{session.Layer}, Tokens: draftTokens, Position: session.Position,
		})
		if err != nil {
			return nil, fmt.Errorf("inference: save %s layer cache: %w", mtpLabel, err)
		}
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	encoder := statecodec.NewEncoder(uint64(math.MaxInt))
	encoder.Raw([]byte(singleHeadMTPStateMagic))
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
		return nil, fmt.Errorf("inference: %s state exceeds addressable memory", mtpLabel)
	}
	return output, nil
}

// LoadMTPSession restores compiled-program-bound single-head draft state.
func (r *Runner) LoadMTPSession(data []byte) (*MTPSession, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	_, _, err := r.singleHeadMTP()
	if err != nil {
		return nil, err
	}
	decoder, err := r.modelStateDecoder(data, singleHeadMTPStateMagic, mtpLabel)
	if err != nil {
		return nil, err
	}
	targetSignature := decoder.Raw(sha256.Size)
	mtpStart := decoder.U32()
	position := decoder.U32()
	trunkLength := decoder.U64()
	layerLength := decoder.U64()
	if decoder.Err() != nil {
		return nil, fmt.Errorf("inference: %s state is truncated", mtpLabel)
	}
	var targetModel [sha256.Size]byte
	copy(targetModel[:], targetSignature)
	if !checked.Nonzero(targetModel) {
		return nil, fmt.Errorf("inference: %s state target binding is invalid", mtpLabel)
	}
	if position < mtpStart || position == math.MaxUint32 ||
		(position == mtpStart) == checked.Nonzero(layerLength) {
		return nil, fmt.Errorf("inference: %s state positions are invalid", mtpLabel)
	}
	hiddenBytes, ok := checked.Bytes(uint64(r.spec.EmbeddingLength), binaryschema.Uint32Bytes)
	if !ok {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", mtpLabel)
	}
	payload := decoder.Remaining()
	if hiddenBytes > payload || trunkLength > payload-hiddenBytes ||
		layerLength != payload-hiddenBytes-trunkLength || !checked.Nonzero(trunkLength) {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", mtpLabel)
	}
	hidden := decodeHiddenState(decoder, uint64(r.spec.EmbeddingLength))
	trunkData := decoder.Raw(trunkLength)
	layerData := decoder.Raw(layerLength)
	if decoder.Done() != nil {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", mtpLabel)
	}
	trunk, err := r.LoadCache(trunkData)
	if err != nil {
		return nil, fmt.Errorf("inference: load %s trunk cache: %w", mtpLabel, err)
	}
	session := &MTPSession{
		TrunkCache: trunk, PendingHidden: hidden, MTPStart: mtpStart,
		Position: position, targetModel: targetModel,
	}
	if checked.Nonzero(layerLength) {
		layerCache, err := unmarshalCache(layerData)
		if err != nil {
			return nil, fmt.Errorf("inference: load %s layer cache: %w", mtpLabel, err)
		}
		layer, found := checked.First(layerCache.Layers)
		if !found || checked.Multiple(len(layerCache.Layers)) || layerCache.Tokens != position-mtpStart ||
			layerCache.Position != position {
			return nil, fmt.Errorf("inference: %s layer cache metadata is invalid", mtpLabel)
		}
		session.Layer = layer
	}
	if err := r.validateMTPSession(session); err != nil {
		return nil, err
	}
	return session, nil
}

func decodeHiddenState(decoder *statecodec.Decoder, width uint64) reference.Value {
	hidden := reference.Value{Shape: tensor.MustShape(width, uint64(tensor.SingletonExtent)), Data: make([]float32, int(width))}
	for index := range hidden.Data {
		hidden.Data[index] = decoder.F32()
	}
	return hidden
}
