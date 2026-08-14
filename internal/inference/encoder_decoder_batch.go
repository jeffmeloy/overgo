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

// EncoderDecoderBatchSession: independent encoder states and decoder caches.
type EncoderDecoderBatchSession struct {
	Sequences     []*EncoderDecoderSession
	SourceLengths []uint32
	SourceWidth   uint32
}

// EncoderDecoderBatchResult: per-sequence unpadded logits.
type EncoderDecoderBatchResult struct {
	Logits  []reference.Value
	Lengths []uint32
}

// NewEncoderDecoderBatchSession: masked padded-source encoding.
func (r *Runner) NewEncoderDecoderBatchSession(
	ctx context.Context,
	batch PaddedTokenBatch,
) (*EncoderDecoderBatchSession, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	width, err := validatePaddedTokenBatch(batch, 0)
	if err != nil {
		return nil, err
	}
	if err := r.lockOpen(); err != nil {
		return nil, err
	}
	defer r.mu.Unlock()
	if r.forwardProgram().Session != model.ForwardSessionEncoderDecoder {
		return nil, errors.New("inference: batch session requires a compiled encoder-decoder program")
	}
	result := &EncoderDecoderBatchSession{
		Sequences:     make([]*EncoderDecoderSession, len(batch.Tokens)),
		SourceLengths: slices.Clone(batch.Lengths),
		SourceWidth:   width,
	}
	for index, row := range batch.Tokens {
		length := batch.Lengths[index]
		encoder, encodeErr := r.forwardEncoderLocked(ctx, row[:length])
		if encodeErr != nil {
			return nil, fmt.Errorf(
				"inference: encoder batch source %d: %w",
				index,
				encodeErr,
			)
		}
		result.Sequences[index] = &EncoderDecoderSession{Encoder: encoder}
	}
	return result, nil
}

// DecodeEncoderDecoderBatch: masked padded-decoder append.
func (r *Runner) DecodeEncoderDecoderBatch(
	ctx context.Context,
	session *EncoderDecoderBatchSession, batch PaddedTokenBatch,
) (EncoderDecoderBatchResult, *EncoderDecoderBatchSession, error) {
	if r == nil {
		return EncoderDecoderBatchResult{}, nil, errors.New("inference: runner is nil")
	}
	if session == nil || len(session.Sequences) == 0 {
		return EncoderDecoderBatchResult{}, nil, errors.New("inference: encoder-decoder batch session is empty")
	}
	if len(session.SourceLengths) != len(session.Sequences) || session.SourceWidth == 0 {
		return EncoderDecoderBatchResult{}, nil, errors.New("inference: encoder batch source mask is incompatible")
	}
	for index, length := range session.SourceLengths {
		if length == 0 || length > session.SourceWidth || session.Sequences[index] == nil {
			return EncoderDecoderBatchResult{}, nil, fmt.Errorf(
				"inference: encoder batch source %d state is incompatible",
				index,
			)
		}
	}
	_, err := validatePaddedTokenBatch(batch, len(session.Sequences))
	if err != nil {
		return EncoderDecoderBatchResult{}, nil, err
	}
	if err := r.lockOpen(); err != nil {
		return EncoderDecoderBatchResult{}, nil, err
	}
	defer r.mu.Unlock()
	if r.forwardProgram().Session != model.ForwardSessionEncoderDecoder {
		return EncoderDecoderBatchResult{}, nil, errors.New("inference: batch decode requires a compiled encoder-decoder program")
	}
	next := &EncoderDecoderBatchSession{
		Sequences:     make([]*EncoderDecoderSession, len(session.Sequences)),
		SourceLengths: slices.Clone(session.SourceLengths),
		SourceWidth:   session.SourceWidth,
	}
	result := EncoderDecoderBatchResult{
		Logits:  make([]reference.Value, len(session.Sequences)),
		Lengths: slices.Clone(batch.Lengths),
	}
	for index, row := range batch.Tokens {
		length := batch.Lengths[index]
		logits, sequence, decodeErr := r.decodeEncoderDecoderLocked(
			ctx,
			session.Sequences[index],
			row[:length],
		)
		if decodeErr != nil {
			return EncoderDecoderBatchResult{}, nil, fmt.Errorf(
				"inference: batch decoder %d: %w",
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
