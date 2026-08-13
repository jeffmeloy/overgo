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

const (
	singleHeadMTPStateHeader = 96
	singleHeadMTPStateMagic  = "L2GMTP02"
)

// SaveMTPSession encodes compiled-program-bound single-head draft state.
func (r *Runner) SaveMTPSession(session *MTPSession) ([]byte, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	plan, _, err := r.singleHeadMTP()
	if err != nil {
		return nil, err
	}
	if err := r.validateMTPSession(session); err != nil {
		return nil, err
	}
	if session.targetModel == [32]byte{} {
		return nil, fmt.Errorf("inference: %s target model binding is missing", plan.Label)
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
			return nil, fmt.Errorf("inference: save %s layer cache: %w", plan.Label, err)
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
		return nil, fmt.Errorf("inference: %s state exceeds addressable memory", plan.Label)
	}
	return output, nil
}

// LoadMTPSession restores compiled-program-bound single-head draft state.
func (r *Runner) LoadMTPSession(data []byte) (*MTPSession, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	plan, _, err := r.singleHeadMTP()
	if err != nil {
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
		return nil, fmt.Errorf("inference: %s state is truncated", plan.Label)
	}
	if string(magic) != singleHeadMTPStateMagic {
		return nil, fmt.Errorf("inference: %s state has invalid magic or version", plan.Label)
	}
	draftModel, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if string(draftSignature) != string(draftModel[:]) {
		return nil, fmt.Errorf("inference: %s state belongs to a different draft model", plan.Label)
	}
	var targetModel [32]byte
	copy(targetModel[:], targetSignature)
	if targetModel == [32]byte{} {
		return nil, fmt.Errorf("inference: %s state target binding is invalid", plan.Label)
	}
	if position < mtpStart || position == math.MaxUint32 ||
		(position == mtpStart) != (layerLength == 0) {
		return nil, fmt.Errorf("inference: %s state positions are invalid", plan.Label)
	}
	hiddenBytes, ok := checked.Bytes(uint64(r.spec.EmbeddingLength), 4)
	if !ok {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", plan.Label)
	}
	payload := decoder.Remaining()
	if hiddenBytes > payload || trunkLength > payload-hiddenBytes ||
		layerLength != payload-hiddenBytes-trunkLength || trunkLength == 0 {
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", plan.Label)
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
		return nil, fmt.Errorf("inference: %s state payload lengths are invalid", plan.Label)
	}
	trunk, err := r.LoadCache(trunkData)
	if err != nil {
		return nil, fmt.Errorf("inference: load %s trunk cache: %w", plan.Label, err)
	}
	session := &MTPSession{
		TrunkCache: trunk, PendingHidden: hidden, MTPStart: mtpStart,
		Position: position, targetModel: targetModel,
	}
	if layerLength > 0 {
		layerCache, err := unmarshalCache(layerData)
		if err != nil {
			return nil, fmt.Errorf("inference: load %s layer cache: %w", plan.Label, err)
		}
		if len(layerCache.Layers) != 1 || layerCache.Tokens != position-mtpStart ||
			layerCache.Position != position {
			return nil, fmt.Errorf("inference: %s layer cache metadata is invalid", plan.Label)
		}
		session.Layer = layerCache.Layers[0]
	}
	if err := r.validateMTPSession(session); err != nil {
		return nil, err
	}
	return session, nil
}
