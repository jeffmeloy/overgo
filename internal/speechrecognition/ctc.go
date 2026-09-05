package speechrecognition

import (
	"context"
	"errors"

	"overgo/internal/checked"
)

// GreedyCTC writes one argmax ID per frame and returns the collapsed prefix of
// tokens. Both destination slices must have one entry per frame and must not
// overlap. Consecutive repetitions are merged before removing blanks; a blank
// therefore separates repetitions. Ties select the lowest vocabulary ID.
func GreedyCTC(ctx context.Context, frameIDs, tokens []int, logits []float32, frames, vocabulary, blank int) ([]int, error) {
	count, ok := checked.MulInt(frames, vocabulary)
	if ctx == nil || frames <= 0 || vocabulary <= 0 || !ok || count != len(logits) || len(frameIDs) != frames || len(tokens) != frames || blank < 0 || blank >= vocabulary || checked.SlicesOverlap(frameIDs, tokens) {
		return nil, errors.New("CTC: invalid context, shape, blank or overlapping destinations")
	}
	previous, n := -1, 0
	for frame := range frames {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		row := logits[frame*vocabulary : (frame+1)*vocabulary]
		best := 0
		for id, value := range row {
			if !checked.Finite32(value) {
				return nil, errors.New("CTC: non-finite logits")
			}
			if value > row[best] {
				best = id
			}
		}
		frameIDs[frame] = best
		if best != previous && best != blank {
			tokens[n] = best
			n++
		}
		previous = best
	}
	return tokens[:n], nil
}
