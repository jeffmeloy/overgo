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
	encoderBytes := uint64(len(session.Encoder.Data)) * 4
	total := uint64(t5SessionHeaderSize) + encoderBytes + uint64(len(cacheData))
	if total > uint64(maxIntValue()) {
		return nil, errors.New("inference: T5 session exceeds addressable memory")
	}
	output := make([]byte, int(total))
	copy(output, t5SessionMagic)
	copy(output[8:40], signature[:])
	binary.LittleEndian.PutUint64(output[40:], session.Encoder.Shape.Dims[0])
	binary.LittleEndian.PutUint64(output[48:], session.Encoder.Shape.Dims[1])
	binary.LittleEndian.PutUint64(output[56:], uint64(len(cacheData)))
	offset := t5SessionHeaderSize
	for _, value := range session.Encoder.Data {
		binary.LittleEndian.PutUint32(output[offset:], math.Float32bits(value))
		offset += 4
	}
	copy(output[offset:], cacheData)
	return output, nil
}

// LoadT5Session: bounded model-bound restore.
func (r *Runner) LoadT5Session(data []byte) (*T5Session, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if len(data) < t5SessionHeaderSize {
		return nil, errors.New("inference: T5 session is truncated")
	}
	if string(data[:8]) != t5SessionMagic {
		return nil, errors.New("inference: T5 session has invalid magic or version")
	}
	signature, err := r.sessionModelSignature()
	if err != nil {
		return nil, err
	}
	if string(data[8:40]) != string(signature[:]) {
		return nil, errors.New("inference: T5 session belongs to a different model")
	}
	width := binary.LittleEndian.Uint64(data[40:])
	tokens := binary.LittleEndian.Uint64(data[48:])
	cacheLength := binary.LittleEndian.Uint64(data[56:])
	if width != uint64(r.spec.EmbeddingLength) || tokens == 0 ||
		(r.spec.ContextLength > 0 && tokens > uint64(r.spec.ContextLength)) {
		return nil, errors.New("inference: T5 session encoder shape is incompatible")
	}
	if width > math.MaxUint64/tokens {
		return nil, errors.New("inference: T5 session encoder size overflows")
	}
	elements := width * tokens
	if elements > math.MaxUint64/4 {
		return nil, errors.New("inference: T5 session encoder byte size overflows")
	}
	encoderBytes := elements * 4
	payload := uint64(len(data) - t5SessionHeaderSize)
	if encoderBytes > payload || cacheLength != payload-encoderBytes {
		return nil, errors.New("inference: T5 session payload lengths are invalid")
	}
	encoder := reference.Value{
		Shape: tensor.MustShape(width, tokens),
		Data:  make([]float32, int(elements)),
	}
	offset := t5SessionHeaderSize
	for index := range encoder.Data {
		encoder.Data[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}
	session := &T5Session{Encoder: encoder}
	if cacheLength > 0 {
		cache, cacheErr := r.LoadCache(data[offset:])
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
	if r.spec.Architecture != "t5" {
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
