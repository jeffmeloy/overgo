//go:build windows

package hybridtrain

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/testcheck"
)

// TestTheFrozenHybridStackGivesItsInputGradient holds the frozen walk on the card to the host reference over a small
// stack of a gated-delta layer and a gated attention layer: with rows placed in place of two tokens' embeddings, the
// masked loss and its gradient with respect to the rows agree, for two draws of rows, and the model is unmoved.
func TestTheFrozenHybridStackGivesItsInputGradient(t *testing.T) {
	// Serial: holds the card.
	cudatest.Require(t)
	const lossTolerance, gradientTolerance = 1e-5, 1e-4
	reference, draw := smallHybridModel(20261107)
	hidden := reference.Geometry.Hidden
	tokens := []int{3, 0, 0, 7, 1, 9, 4}
	trained := []bool{false, false, false, false, true, true, true}
	positions := []int{1, 2}
	geometry := reference.Geometry
	geometry.Tokens = len(tokens)
	head := slices.Clone(reference.Head)
	model := &FrozenLanguageModel{
		Geometry: geometry, Layers: reference.Layers, FinalNorm: reference.FinalNorm, Head: reference.Head, Vocab: reference.Vocab,
		rows: func(_ context.Context, ids []uint32) ([]float32, error) {
			picked := make([]int, len(ids))
			for index, id := range ids {
				picked[index] = int(id)
			}
			rows := make([]float32, len(ids)*hidden)
			hostmath.GatherRows(rows, reference.Embedding, picked, hidden)
			return rows, nil
		},
	}
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	session := testcheck.Must(OpenFrozenSession(t.Context(), model, worker, len(tokens)))(t)
	for draws := range 2 {
		rows := draw(len(positions) * hidden)
		loss, gradient, err := session.Run(t.Context(), tokens, trained, positions, rows)
		testcheck.NoError(t, err)
		wantLoss, wantGradient, err := reference.maskedLossAndInputGradient(tokens, trained, positions, rows)
		testcheck.NoError(t, err)
		var scale, furthest float64
		for index, value := range wantGradient {
			scale = max(scale, math.Abs(float64(value)))
			furthest = max(furthest, math.Abs(float64(gradient[index]-value)))
		}
		t.Logf("draw %d: loss %.7f on the card, %.7f on the host; the row gradients stand %.3g apart, the host's largest %.3g", draws, loss, wantLoss, furthest, scale)
		if math.Abs(loss-wantLoss) > lossTolerance*max(1, wantLoss) || furthest > gradientTolerance*scale {
			t.Errorf("draw %d: the card's loss %v and gradient stand from the host's %v by %.3g of its largest %.3g", draws, loss, wantLoss, furthest, scale)
		}
	}
	testcheck.NoError(t, errors.Join(session.Close()))
	if !slices.Equal(head, reference.Head) {
		t.Error("the frozen walk changed the model's weights")
	}
}
