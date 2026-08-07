package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// PaddedTokenBatch: rectangular tokens plus unpadded lengths.
type PaddedTokenBatch struct {
	Tokens  [][]tokenizer.TokenID
	Lengths []uint32
}

// T5BatchSession: independent encoder states and decoder caches.
type T5BatchSession struct {
	Sequences     []*T5Session
	SourceLengths []uint32
	SourceWidth   uint32
}

// T5BatchResult: per-sequence unpadded logits.
type T5BatchResult struct {
	Logits  []reference.Value
	Lengths []uint32
}

// NewT5BatchSession: masked padded-source encoding.
func (r *Runner) NewT5BatchSession(
	ctx context.Context,
	batch PaddedTokenBatch,
) (*T5BatchSession, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	width, err := validatePaddedTokenBatch(batch, 0)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: runner is closed")
	}
	if r.forwardPolicy() != model.ForwardT5 {
		return nil, errors.New("inference: T5 batch session requires T5 architecture")
	}
	result := &T5BatchSession{
		Sequences:     make([]*T5Session, len(batch.Tokens)),
		SourceLengths: slices.Clone(batch.Lengths),
		SourceWidth:   width,
	}
	for index, row := range batch.Tokens {
		length := batch.Lengths[index]
		encoder, encodeErr := r.forwardT5EncoderLocked(ctx, row[:length])
		if encodeErr != nil {
			return nil, fmt.Errorf(
				"inference: T5 batch source %d: %w",
				index,
				encodeErr,
			)
		}
		result.Sequences[index] = &T5Session{Encoder: encoder}
	}
	return result, nil
}

// DecodeT5Batch: masked padded-decoder append.
func (r *Runner) DecodeT5Batch(
	ctx context.Context,
	session *T5BatchSession,
	batch PaddedTokenBatch,
) (T5BatchResult, *T5BatchSession, error) {
	if r == nil {
		return T5BatchResult{}, nil, errors.New("inference: runner is nil")
	}
	if session == nil || len(session.Sequences) == 0 {
		return T5BatchResult{}, nil, errors.New("inference: T5 batch session is empty")
	}
	if len(session.SourceLengths) != len(session.Sequences) || session.SourceWidth == 0 {
		return T5BatchResult{}, nil, errors.New("inference: T5 batch source mask is incompatible")
	}
	for index, length := range session.SourceLengths {
		if length == 0 || length > session.SourceWidth || session.Sequences[index] == nil {
			return T5BatchResult{}, nil, fmt.Errorf(
				"inference: T5 batch source %d state is incompatible",
				index,
			)
		}
	}
	_, err := validatePaddedTokenBatch(batch, len(session.Sequences))
	if err != nil {
		return T5BatchResult{}, nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return T5BatchResult{}, nil, errors.New("inference: runner is closed")
	}
	if r.forwardPolicy() != model.ForwardT5 {
		return T5BatchResult{}, nil, errors.New("inference: T5 batch decode requires T5 architecture")
	}
	next := &T5BatchSession{
		Sequences:     make([]*T5Session, len(session.Sequences)),
		SourceLengths: slices.Clone(session.SourceLengths),
		SourceWidth:   session.SourceWidth,
	}
	result := T5BatchResult{
		Logits:  make([]reference.Value, len(session.Sequences)),
		Lengths: slices.Clone(batch.Lengths),
	}
	for index, row := range batch.Tokens {
		length := batch.Lengths[index]
		logits, sequence, decodeErr := r.decodeT5Locked(
			ctx,
			session.Sequences[index],
			row[:length],
		)
		if decodeErr != nil {
			return T5BatchResult{}, nil, fmt.Errorf(
				"inference: T5 batch decoder %d: %w",
				index,
				decodeErr,
			)
		}
		result.Logits[index] = logits
		next.Sequences[index] = sequence
	}
	return result, next, nil
}

func validatePaddedTokenBatch(
	batch PaddedTokenBatch,
	wantRows int,
) (uint32, error) {
	if len(batch.Tokens) == 0 {
		return 0, errors.New("inference: padded token batch is empty")
	}
	if wantRows > 0 && len(batch.Tokens) != wantRows {
		return 0, fmt.Errorf(
			"inference: padded token batch has %d rows, need %d",
			len(batch.Tokens),
			wantRows,
		)
	}
	if len(batch.Lengths) != len(batch.Tokens) {
		return 0, errors.New("inference: padded token batch length count differs")
	}
	width := len(batch.Tokens[0])
	if width == 0 {
		return 0, errors.New("inference: padded token batch width is zero")
	}
	for index, row := range batch.Tokens {
		if len(row) != width {
			return 0, fmt.Errorf(
				"inference: padded token batch row %d has width %d, need %d",
				index,
				len(row),
				width,
			)
		}
		if batch.Lengths[index] == 0 || batch.Lengths[index] > uint32(width) {
			return 0, fmt.Errorf(
				"inference: padded token batch row %d length %d is invalid",
				index,
				batch.Lengths[index],
			)
		}
	}
	return uint32(width), nil
}
