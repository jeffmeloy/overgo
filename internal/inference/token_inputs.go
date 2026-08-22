package inference

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/tokenizer"
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
	rows, err := r.vocab.TensorIndices(tokenIDs)
	if err != nil {
		return forwardSequencePlan{}, err
	}
	return r.planForwardIndices(rows, pastTokens, nextPosition)
}

func (r *Runner) planForwardIndices(
	rows []uint32,
	pastTokens, nextPosition uint32,
) (forwardSequencePlan, error) {
	totalTokens, validTotal := checked.Add64(uint64(pastTokens), uint64(len(rows)))
	if !validTotal || !checked.AtMost64(totalTokens, uint64(r.spec.ContextLength)) {
		return forwardSequencePlan{}, fmt.Errorf(
			"inference: cached plus new token count %d exceeds context length %d",
			totalTokens, r.spec.ContextLength,
		)
	}
	absolutePosition, validPosition := checked.Add64(uint64(nextPosition), uint64(len(rows)))
	if !validPosition || !checked.AtMost64(absolutePosition, math.MaxUint32) {
		return forwardSequencePlan{}, errors.New("inference: absolute token position exceeds uint32")
	}
	return forwardSequencePlan{
		rows: rows, positions: tokenPositions(nextPosition, len(rows)),
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

func tokenPositions(start uint32, count int) []uint32 {
	positions := make([]uint32, count)
	for index := range positions {
		positions[index] = start + uint32(index)
	}
	return positions
}
