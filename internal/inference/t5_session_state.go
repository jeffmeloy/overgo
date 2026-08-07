package inference

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/statecodec"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const (
	t5SessionMagic      = "L2GT5S01"
	t5SessionHeaderSize = 64
)

// SaveT5Session: model-bound encoder state and decoder cache.
func (r *Runner) SaveT5Session(session *T5Session) ([]byte, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if err := r.validateT5Session(session); err != nil {
		return nil, err
	}
	var cacheData []byte
	var err error
	if session.Cache != nil {
		cacheData, err = r.SaveCache(session.Cache)
		if err != nil {
			return nil, err
		}
	}
	signature, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	encoder := statecodec.NewEncoder(uint64(math.MaxInt))
	encoder.Raw([]byte(t5SessionMagic))
	encoder.Raw(signature[:])
	encoder.U64(session.Encoder.Shape.Dims[0])
	encoder.U64(session.Encoder.Shape.Dims[1])
	encoder.U64(uint64(len(cacheData)))
	for _, value := range session.Encoder.Data {
		encoder.F32(value)
	}
	encoder.Raw(cacheData)
	output, err := encoder.Data()
	if err != nil {
		return nil, errors.New("inference: T5 session exceeds addressable memory")
	}
	return output, nil
}

// LoadT5Session: bounded model-bound restore.
func (r *Runner) LoadT5Session(data []byte) (*T5Session, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	decoder := statecodec.NewDecoder(data, uint64(math.MaxInt))
	magic := decoder.Raw(8)
	modelSignature := decoder.Raw(32)
	width := decoder.U64()
	tokens := decoder.U64()
	cacheLength := decoder.U64()
	if decoder.Err() != nil {
		return nil, errors.New("inference: T5 session is truncated")
	}
	if string(magic) != t5SessionMagic {
		return nil, errors.New("inference: T5 session has invalid magic or version")
	}
	signature, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if string(modelSignature) != string(signature[:]) {
		return nil, errors.New("inference: T5 session belongs to a different model")
	}
	if width != uint64(r.spec.EmbeddingLength) || tokens == 0 ||
		(r.spec.ContextLength > 0 && tokens > uint64(r.spec.ContextLength)) {
		return nil, errors.New("inference: T5 session encoder shape is incompatible")
	}
	elements, ok := checked.Mul64(width, tokens)
	if !ok {
		return nil, errors.New("inference: T5 session encoder size overflows")
	}
	encoderBytes, ok := checked.Bytes(elements, 4)
	if !ok {
		return nil, errors.New("inference: T5 session encoder byte size overflows")
	}
	payload := decoder.Remaining()
	if encoderBytes > payload || cacheLength != payload-encoderBytes {
		return nil, errors.New("inference: T5 session payload lengths are invalid")
	}
	encoder := reference.Value{
		Shape: tensor.MustShape(width, tokens),
		Data:  make([]float32, int(elements)),
	}
	for index := range encoder.Data {
		encoder.Data[index] = decoder.F32()
	}
	cacheData := decoder.Raw(cacheLength)
	if decoder.Done() != nil {
		return nil, errors.New("inference: T5 session payload lengths are invalid")
	}
	session := &T5Session{Encoder: encoder}
	if cacheLength > 0 {
		cache, cacheErr := r.LoadCache(cacheData)
		if cacheErr != nil {
			return nil, fmt.Errorf("inference: load T5 decoder cache: %w", cacheErr)
		}
		session.Cache = cache
	}
	if err := r.validateT5Session(session); err != nil {
		return nil, err
	}
	return session, nil
}

func (r *Runner) validateT5Session(session *T5Session) error {
	if r.forwardPolicy() != model.ForwardT5 {
		return errors.New("inference: T5 session requires T5 architecture")
	}
	if session == nil || session.Encoder.Shape.Rank != 2 ||
		session.Encoder.Shape.Dims[0] != uint64(r.spec.EmbeddingLength) ||
		session.Encoder.Shape.Dims[1] == 0 {
		return errors.New("inference: T5 encoder state shape is incompatible")
	}
	if r.spec.ContextLength > 0 && session.Encoder.Shape.Dims[1] > uint64(r.spec.ContextLength) {
		return errors.New("inference: T5 encoder state exceeds context length")
	}
	if err := validateStateValue(session.Encoder); err != nil {
		return fmt.Errorf("inference: T5 encoder state: %w", err)
	}
	if session.Cache != nil {
		return r.validateT5Cache(session.Cache, session.Encoder.Shape.Dims[1])
	}
	return nil
}
