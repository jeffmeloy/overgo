package thoughtbank

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/sequencescore"
)

// The model's only memory of anything before the window in front of it is its
// bank: after each chunk of a document it writes one note into it. Its own
// code measures what that memory is worth by a turn on which the model is
// shown nothing: the input is blank tokens, and each position is asked for the
// token at that place in the opening of the next chunk. The loss of that turn
// with the notes of the chunks before it, against the loss with a fresh bank,
// is what the notes carried. Scoring a passage window by window from an empty
// bank, which is what a perplexity reads, never writes or reads a note.

// memoryTokens is how many tokens the model's own code adds after its
// tokenizer's vocabulary, the mark that closes a chunk it is to take a note of
// and the blank a deferred turn is filled with, in that order.
const memoryTokens = 2

// ScoreCarried reads the first chunks windows of text into the bank, each
// closed with the model's note mark and each written from the bank the one
// before left, then scores the first opening tokens of the next window on a
// turn of blank input: carried, from that bank, and reset, from a fresh one.
// A fresh bank is the served one, seed slots of zeros. A text too short to
// hold the chunks and the opening is refused.
func (g *Generator) ScoreCarried(ctx context.Context, text string, chunks, opening int) (carried, reset sequencescore.Score, err error) {
	if g == nil || g.weights == nil || g.tokenizer == nil {
		return carried, reset, errors.New("thoughtbank: generator is unavailable")
	}
	if chunks <= 0 || opening <= 0 || opening > scoringWindowTokens {
		return carried, reset, fmt.Errorf("thoughtbank: %d chunks carried to an opening of %d tokens is not a deferred turn", chunks, opening)
	}
	encoded, err := g.tokenizer.Encode(text)
	if err != nil {
		return carried, reset, err
	}
	written := chunks * scoringWindowTokens
	if len(encoded) < written+opening {
		return carried, reset, fmt.Errorf("thoughtbank: a text of %d tokens holds no %d chunks of %d and an opening of %d after them", len(encoded), chunks, scoringWindowTokens, opening)
	}
	think, blank := int32(g.weights.VocabSize-memoryTokens), int32(g.weights.VocabSize-1)
	seeds := g.config.MemSeedSlots
	bank, slots := make([]float32, seeds*g.config.MemDim), seeds
	for chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return carried, reset, err
		}
		ids := make([]int32, 0, scoringWindowTokens+1)
		for _, token := range encoded[chunk*scoringWindowTokens : (chunk+1)*scoringWindowTokens] {
			ids = append(ids, int32(token))
		}
		output, err := FastWeightBankLMForward(append(ids, think), bank, slots, g.weights)
		if err != nil {
			return carried, reset, err
		}
		bank, slots = output.MemBank, output.Slots
	}
	blanks := make([]int32, opening)
	for position := range blanks {
		blanks[position] = blank
	}
	deferred := func(bank []float32, slots int) (sequencescore.Score, error) {
		if err := ctx.Err(); err != nil {
			return sequencescore.Score{}, err
		}
		output, err := FastWeightBankLMForward(blanks, bank, slots, g.weights)
		if err != nil {
			return sequencescore.Score{}, err
		}
		score := sequencescore.Score{Tokens: uint64(opening)}
		vocabulary := g.weights.VocabSize
		for position, token := range encoded[written : written+opening] {
			score.LogProbability += logProbability(output.Logits[position*vocabulary:(position+1)*vocabulary], token)
		}
		return score, nil
	}
	if carried, err = deferred(bank, slots); err != nil {
		return carried, reset, err
	}
	reset, err = deferred(make([]float32, seeds*g.config.MemDim), seeds)
	return carried, reset, err
}
