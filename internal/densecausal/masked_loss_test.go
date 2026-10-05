package densecausal

import (
	"math"
	"slices"
	"testing"
)

// maskedTokens is a window of eight tokens whose first four are the prompt a
// run reads and whose last four are the reply it learns.
var (
	maskedTokens  = []int{3, 9, 14, 5, 21, 7, 18, 2}
	maskedTrained = []bool{false, false, false, false, true, true, true, true}
)

// positionLosses is each position's cross-entropy against the token that
// follows it, from the model's own logits, in float64.
func positionLosses(t *testing.T, model *Model, tokens []int) []float64 {
	t.Helper()
	_, logits, err := model.Loss(tokens)
	if err != nil {
		t.Fatal(err)
	}
	vocabulary := model.Dims.Vocab
	losses := make([]float64, len(tokens)-1)
	for position := range losses {
		row := logits[position*vocabulary : (position+1)*vocabulary]
		peak := float64(slices.Max(row))
		var sum float64
		for _, value := range row {
			sum += math.Exp(float64(value) - peak)
		}
		losses[position] = peak + math.Log(sum) - float64(row[tokens[position+1]])
	}
	return losses
}

// TestMaskedLossNormalizesOverTrainedTokensOnly checks the masked loss is the
// mean cross-entropy of the trained targets, whatever else the window holds,
// and that learning every token is what LossAndGrads already computes.
func TestMaskedLossNormalizesOverTrainedTokensOnly(t *testing.T) {
	model := tinyMuonModel(t)
	losses := positionLosses(t, model, maskedTokens)
	var want float64
	var learned int
	for position, loss := range losses {
		if maskedTrained[position+1] {
			want += loss
			learned++
		}
	}
	want /= float64(learned)
	got, _, err := model.MaskedLossAndGrads(maskedTokens, maskedTrained)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-want) > 1e-5 {
		t.Fatalf("masked loss %.7f, mean of the trained targets' losses %.7f", got, want)
	}
	if every := mean(losses); math.Abs(got-every) < 1e-3 {
		t.Fatalf("masked loss %.7f equals the loss over every token %.7f: the mask changed nothing", got, every)
	}
	whole, _, wholeGrads, err := model.LossAndGrads(maskedTokens)
	if err != nil {
		t.Fatal(err)
	}
	all := make([]bool, len(maskedTokens))
	for index := range all {
		all[index] = true
	}
	flagged, flaggedGrads, err := model.MaskedLossAndGrads(maskedTokens, all)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(whole-flagged) > 1e-5 {
		t.Fatalf("every token flagged: loss %.7f, LossAndGrads %.7f", flagged, whole)
	}
	for name, grad := range wholeGrads {
		for index, value := range grad {
			if math.Abs(float64(value-flaggedGrads[name][index])) > 1e-5 {
				t.Fatalf("every token flagged: %s[%d] gradient %g, LossAndGrads %g", name, index, flaggedGrads[name][index], value)
			}
		}
	}
}

func mean(values []float64) float64 {
	var sum float64
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

// TestMaskedTokenHasNoGradient checks a token that is not trained, and that
// nothing after it reads, changes neither the loss nor any gradient: the last
// token of the window is only ever a target, so swapping it for another must
// leave the step exactly as it was while its flag is down, and move it once
// the flag is up.
func TestMaskedTokenHasNoGradient(t *testing.T) {
	model := tinyMuonModel(t)
	other := slices.Clone(maskedTokens)
	other[len(other)-1] = (other[len(other)-1] + 5) % model.Dims.Vocab
	masked := slices.Clone(maskedTrained)
	masked[len(masked)-1] = false
	loss, grads, err := model.MaskedLossAndGrads(maskedTokens, masked)
	if err != nil {
		t.Fatal(err)
	}
	swappedLoss, swappedGrads, err := model.MaskedLossAndGrads(other, masked)
	if err != nil {
		t.Fatal(err)
	}
	if loss != swappedLoss {
		t.Fatalf("a masked final token moved the loss: %.9f against %.9f", loss, swappedLoss)
	}
	for name, grad := range grads {
		if !slices.Equal(grad, swappedGrads[name]) {
			t.Fatalf("a masked final token moved the gradient of %s", name)
		}
	}
	_, learnedGrads, err := model.MaskedLossAndGrads(maskedTokens, maskedTrained)
	if err != nil {
		t.Fatal(err)
	}
	_, learnedSwapped, err := model.MaskedLossAndGrads(other, maskedTrained)
	if err != nil {
		t.Fatal(err)
	}
	moved := false
	for name, grad := range learnedGrads {
		moved = moved || !slices.Equal(grad, learnedSwapped[name])
	}
	if !moved {
		t.Fatal("a trained final token moved no gradient: the swap proves nothing")
	}
}

// TestMaskedLossRefusesWhatItCannotLearn checks a window of no trained token,
// and flags that do not match the tokens, are errors rather than a loss of
// zero over nothing.
func TestMaskedLossRefusesWhatItCannotLearn(t *testing.T) {
	model := tinyMuonModel(t)
	none := make([]bool, len(maskedTokens))
	none[0] = true // the first token is predicted by nothing: its flag does not count
	if _, _, err := model.MaskedLossAndGrads(maskedTokens, none); err == nil {
		t.Fatal("a window with no trained target was accepted")
	}
	if _, _, err := model.MaskedLossAndGrads(maskedTokens, maskedTrained[:3]); err == nil {
		t.Fatal("flags for three of eight tokens were accepted")
	}
}

// TestTrainWithMaskStepsOnTheTrainedTokens checks a masked training run, with
// every token flagged, takes the steps Train takes, and with the prompt
// masked takes others.
func TestTrainWithMaskStepsOnTheTrainedTokens(t *testing.T) {
	const rate, momentum = 0.01, 0.9
	batches := [][]int{maskedTokens, maskedTokens}
	all := make([]bool, len(maskedTokens))
	for index := range all {
		all[index] = true
	}
	baseline, _, err := tinyMuonModel(t).Train(batches, rate, momentum, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	flagged, _, err := tinyMuonModel(t).TrainWith(batches, rate, momentum, nil, nil, TrainOptions{Trained: [][]bool{all, all}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(baseline, flagged) {
		t.Fatalf("every token flagged: losses %v, Train's %v", flagged, baseline)
	}
	masked, _, err := tinyMuonModel(t).TrainWith(batches, rate, momentum, nil, nil, TrainOptions{Trained: [][]bool{maskedTrained, maskedTrained}})
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(masked, baseline) {
		t.Fatalf("a masked run took Train's steps: losses %v", masked)
	}
	if _, _, err := tinyMuonModel(t).TrainWith(batches, rate, momentum, nil, nil, TrainOptions{Trained: [][]bool{all}}); err == nil {
		t.Fatal("flags for one of two batches were accepted")
	}
}
