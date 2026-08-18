package trainingprogram

import (
	"math"
	"testing"

	"overgo/internal/sequencescore"
)

func continuationScore(value float64) sequencescore.Score {
	return sequencescore.Score{LogProbability: value, Tokens: 1}
}

func TestContinuationScoreSharedAcrossEvaluationPerplexityAndDPO(t *testing.T) {
	chosen, err := sequencescore.Selected([]float64{-5, -2}, []bool{false, true})
	if err != nil {
		t.Fatal(err)
	}
	rejected, err := sequencescore.Selected([]float64{-5, -3}, []bool{false, true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := DPOLoss(PreferenceScores{
		PolicyChosen: chosen, PolicyRejected: rejected,
		ReferenceChosen: chosen, ReferenceRejected: rejected,
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if chosen.Tokens != rejected.Tokens || result.Loss != math.Log(2) {
		t.Fatalf("continuation/DPO result = %+v/%+v/%+v", chosen, rejected, result)
	}
}

func TestDPOLossMatchesClosedForm(t *testing.T) {
	scores := PreferenceScores{
		PolicyChosen: continuationScore(-1), PolicyRejected: continuationScore(-3),
		ReferenceChosen: continuationScore(-2), ReferenceRejected: continuationScore(-3),
	}
	const scale = 0.25
	got, err := DPOLoss(scores, scale)
	if err != nil {
		t.Fatal(err)
	}
	margin := scale * ((scores.PolicyChosen.LogProbability - scores.PolicyRejected.LogProbability) -
		(scores.ReferenceChosen.LogProbability - scores.ReferenceRejected.LogProbability))
	want := math.Log1p(math.Exp(-margin))
	if math.Abs(got.Loss-want) > 1e-15 {
		t.Fatalf("loss=%g want=%g", got.Loss, want)
	}
}

func TestDPOGradientFiniteDifference(t *testing.T) {
	scores := PreferenceScores{
		PolicyChosen: continuationScore(-2), PolicyRejected: continuationScore(-1),
		ReferenceChosen: continuationScore(-1.5), ReferenceRejected: continuationScore(-2.5),
	}
	const scale, epsilon = 0.4, 1e-6
	result, err := DPOLoss(scores, scale)
	if err != nil {
		t.Fatal(err)
	}
	plus, minus := scores, scores
	plus.PolicyChosen.LogProbability += epsilon
	minus.PolicyChosen.LogProbability -= epsilon
	plusLoss, _ := DPOLoss(plus, scale)
	minusLoss, _ := DPOLoss(minus, scale)
	want := (plusLoss.Loss - minusLoss.Loss) / (2 * epsilon)
	if math.Abs(result.ChosenGradient-want) > 1e-9 {
		t.Fatalf("chosen gradient=%g want=%g", result.ChosenGradient, want)
	}
}

func TestDPOPreferredUpdateDirection(t *testing.T) {
	zero := continuationScore(0)
	result, err := DPOLoss(PreferenceScores{
		PolicyChosen: zero, PolicyRejected: zero, ReferenceChosen: zero, ReferenceRejected: zero,
	}, 1)
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
		{scores: PreferenceScores{PolicyChosen: continuationScore(math.NaN())}, scale: 1},
	} {
		if _, err := DPOLoss(input.scores, input.scale); err == nil {
			t.Fatalf("accepted scores=%+v scale=%g", input.scores, input.scale)
		}
	}
}
