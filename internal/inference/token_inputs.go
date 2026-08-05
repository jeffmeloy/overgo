package inference

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tokenizer"
)

type forwardSequencePlan struct {
	rows         []uint32
	positions    []uint32
	pastTokens   uint32
	nextPosition uint32
}

func (r *Runner) planForwardSequence(
	tokenIDs []tokenizer.TokenID,
	pastTokens, nextPosition uint32,
) (forwardSequencePlan, error) {
	if uint64(pastTokens)+uint64(len(tokenIDs)) > uint64(r.spec.ContextLength) {
		return forwardSequencePlan{}, fmt.Errorf(
			"inference: cached plus new token count %d exceeds context length %d",
			uint64(pastTokens)+uint64(len(tokenIDs)), r.spec.ContextLength,
		)
	}
	if uint64(nextPosition)+uint64(len(tokenIDs)) > math.MaxUint32 {
		return forwardSequencePlan{}, errors.New("inference: absolute token position exceeds uint32")
	}
	rows, err := r.tokenRows(tokenIDs)
	if err != nil {
		return forwardSequencePlan{}, err
	}
	return forwardSequencePlan{
		rows: rows, positions: tokenPositions(nextPosition, len(tokenIDs)),
		pastTokens: pastTokens, nextPosition: nextPosition,
	}, nil
}

func validateLearnedPositions(positions []uint32, contextLength uint32) error {
	for _, position := range positions {
		if position >= contextLength {
			return fmt.Errorf(
				"inference: learned position %d exceeds context length %d", position, contextLength,
			)
		}
	}
	return nil
}

func (r *Runner) tokenRows(tokenIDs []tokenizer.TokenID) ([]uint32, error) {
	rows := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return nil, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
	}
	return rows, nil
}

func tokenPositions(start uint32, count int) []uint32 {
	positions := make([]uint32, count)
	for index := range positions {
		positions[index] = start + uint32(index)
	}
	return positions
}
