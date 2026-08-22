package inference

import (
	"context"
	"errors"
	"fmt"
	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// EncoderDecoderBatchSession: independent encoder states and decoder caches.
type EncoderDecoderBatchSession struct {
	Sequences []*EncoderDecoderSession
	Source    tokenizer.PaddedBatchLayout
}

// EncoderDecoderBatchResult: per-sequence unpadded logits.
type EncoderDecoderBatchResult struct {
	Logits  []reference.Value
	Lengths []uint32
}

// NewEncoderDecoderBatchSession: masked padded-source encoding.
func (r *Runner) NewEncoderDecoderBatchSession(
	ctx context.Context,
	batch tokenizer.PaddedBatch,
) (*EncoderDecoderBatchSession, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	layout, err := batch.Layout(0)
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
		Sequences: make([]*EncoderDecoderSession, len(batch.Tokens)),
		Source:    layout,
	}
	for index, row := range batch.Tokens {
		length := layout.Lengths[index]
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
	session *EncoderDecoderBatchSession, batch tokenizer.PaddedBatch,
) (EncoderDecoderBatchResult, *EncoderDecoderBatchSession, error) {
	if r == nil {
		return EncoderDecoderBatchResult{}, nil, errRunnerNil
	}
	if session == nil || len(session.Sequences) == 0 {
		return EncoderDecoderBatchResult{}, nil, errors.New("inference: encoder-decoder batch session is empty")
	}
	if !session.Source.Valid(len(session.Sequences)) {
		return EncoderDecoderBatchResult{}, nil, errors.New("inference: encoder batch source mask is incompatible")
	}
	for index, sequence := range session.Sequences {
		if sequence == nil {
			return EncoderDecoderBatchResult{}, nil, fmt.Errorf(
				"inference: encoder batch source %d state is incompatible",
				index,
			)
		}
	}
	layout, err := batch.Layout(len(session.Sequences))
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
		Sequences: make([]*EncoderDecoderSession, len(session.Sequences)),
		Source:    session.Source.Clone(),
	}
	result := EncoderDecoderBatchResult{
		Logits:  make([]reference.Value, len(session.Sequences)),
		Lengths: layout.Clone().Lengths,
	}
	for index, row := range batch.Tokens {
		length := layout.Lengths[index]
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
