package inference

import (
	"fmt"

	"llamacpp2go/internal/tokenizer"
)

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
