package densecausal

import (
	"errors"
	"fmt"

	"overgo/internal/hostmath"
)

// MaskedLossAndInputGradient is the model's loss over the trained tokens with rows put in place of the embeddings of
// the tokens at positions, every weight frozen, and the loss's gradient with respect to each of those rows: a frozen
// text model's backward to an adapter that feeds its input, as an image composite's bridge does. A row predicts the
// token after it, as MaskedLossAndLayerGradsOf's do.
func (m *Model) MaskedLossAndInputGradient(tokens []int, trained []bool, positions []int, rows []float32) (float64, []float32, error) {
	hidden := m.Dims.Hidden
	if trained == nil || len(rows) != len(positions)*hidden {
		return 0, nil, fmt.Errorf("densecausal: trained flags and %d values for %d placed rows", len(rows), len(positions))
	}
	for _, position := range positions {
		if position < 0 || position >= len(tokens) {
			return 0, nil, errors.New("densecausal: a placed row stands outside the tokens")
		}
	}
	gradient := make([]float32, len(rows))
	loss, _, _, err := m.lossAndGradsSettled(tokens, trained, false, nil, nil, nil, func(x []float32) {
		hostmath.ScatterRows(x, rows, positions, hidden)
	}, func(input []float32) {
		hostmath.GatherRows(gradient, input, positions, hidden)
	})
	return loss, gradient, err
}
