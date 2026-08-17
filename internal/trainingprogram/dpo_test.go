package trainingprogram

import (
	"math"
	"testing"
)

func TestDPOLossMatchesClosedForm(t *testing.T) {
	scores := PreferenceScores{PolicyChosen: -1, PolicyRejected: -3, ReferenceChosen: -2, ReferenceRejected: -3}
	const scale = 0.25
	got, err := DPOLoss(scores, scale)
	if err != nil {
		t.Fatal(err)
	}
	margin := scale * ((scores.PolicyChosen - scores.PolicyRejected) - (scores.ReferenceChosen - scores.ReferenceRejected))
	want := math.Log1p(math.Exp(-margin))
	if math.Abs(got.Loss-want) > 1e-15 {
		t.Fatalf("loss=%g want=%g", got.Loss, want)
	}
}

func TestDPOGradientFiniteDifference(t *testing.T) {
	scores := PreferenceScores{PolicyChosen: -2, PolicyRejected: -1, ReferenceChosen: -1.5, ReferenceRejected: -2.5}
	const scale, epsilon = 0.4, 1e-6
	result, err := DPOLoss(scores, scale)
	if err != nil {
		t.Fatal(err)
	}
	plus, minus := scores, scores
	plus.PolicyChosen += epsilon
	minus.PolicyChosen -= epsilon
	plusLoss, _ := DPOLoss(plus, scale)
	minusLoss, _ := DPOLoss(minus, scale)
	want := (plusLoss.Loss - minusLoss.Loss) / (2 * epsilon)
	if math.Abs(result.ChosenGradient-want) > 1e-9 {
		t.Fatalf("chosen gradient=%g want=%g", result.ChosenGradient, want)
	}
}

func TestDPOPreferredUpdateDirection(t *testing.T) {
	result, err := DPOLoss(PreferenceScores{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if result.ChosenGradient >= 0 || result.RejectedGradient <= 0 {
		t.Fatalf("score gradients=%+v", result)
	}
}

func TestDPORejectsInvalidInput(t *testing.T) {
	for _, input := range []struct {
		scores PreferenceScores
		scale  float64
	}{
		{scale: 0},
		{scale: math.Inf(1)},
		{scores: PreferenceScores{PolicyChosen: math.NaN()}, scale: 1},
	} {
		if _, err := DPOLoss(input.scores, input.scale); err == nil {
			t.Fatalf("accepted scores=%+v scale=%g", input.scores, input.scale)
		}
	}
}
