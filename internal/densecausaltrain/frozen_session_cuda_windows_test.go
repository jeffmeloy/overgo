//go:build windows

package densecausaltrain

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/densecausal"
	"overgo/internal/testcheck"
)

// TestFrozenSessionMatchesTheHostInputGradient holds the device's frozen run to the host's: rows placed at positions
// give the masked loss and its gradient with respect to the rows the host's frozen backward gives, for a model with
// attention biases as Qwen2.5's, and the model's weights are what they were.
func TestFrozenSessionMatchesTheHostInputGradient(t *testing.T) {
	cudatest.Require(t)
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	for name, model := range map[string]*densecausal.Model{"tied": tinyMuonModel(t), "qwen2 bias": modelFromGolden(t, readQwen2Golden(t))} {
		tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
		trained := []bool{false, false, false, false, true, true, true, true}
		positions := []int{1, 2}
		draws := rand.New(rand.NewPCG(3, 7))
		rows := make([]float32, len(positions)*model.Dims.Hidden)
		for index := range rows {
			rows[index] = float32(draws.NormFloat64() * 0.5)
		}
		embedding, _, _ := model.LexicalTensorNames()
		before := slices.Clone(model.Weights[embedding])
		wantLoss, wantGradient, err := model.MaskedLossAndInputGradient(tokens, trained, positions, rows)
		testcheck.NoError(t, err)
		session := testcheck.Must(OpenFrozenSession(model, worker, len(tokens)))(t)
		loss, gradient, err := session.Run(tokens, trained, positions, rows)
		testcheck.NoError(t, errors.Join(err, session.Close()))
		if delta := math.Abs(loss - wantLoss); delta > 2e-4 {
			t.Errorf("%s: device loss %v, host %v", name, loss, wantLoss)
		}
		var scale, furthest float64
		for index, value := range wantGradient {
			scale = max(scale, math.Abs(float64(value)))
			furthest = max(furthest, math.Abs(float64(gradient[index]-value)))
		}
		if furthest > 1e-2*scale {
			t.Errorf("%s: device row gradient stands %.3g from the host's, whose largest is %.3g", name, furthest, scale)
		}
		if !slices.Equal(before, model.Weights[embedding]) {
			t.Errorf("%s: the frozen run changed the model's weights", name)
		}
	}
}
