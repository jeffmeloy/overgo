package trainingprogram

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
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

// TestSimPOLossMatchesHandComputedValues checks SimPO's loss (Meng, Xia and
// Chen 2024), -log σ(β/|yw|·log π(yw|x) - β/|yl|·log π(yl|x) - γ), on a
// two-token chosen and a three-token rejected completion. By hand, with
// log π(yw|x) = -2, log π(yl|x) = -6, β = 2 and γ = 1: the reward margin is
// 2·(-2/2 + 6/3) - 1 = 1 and the loss log(1 + e^-1) = 0.31326168751822286.
// With σ(-1) = 1/(1 + e) = 0.2689414213699951, the gradients with respect
// to the summed log probabilities are -β·σ(-1)/2 = -0.2689414213699951 and
// β·σ(-1)/3 = 0.17929428091333008. The loss reads no reference, so the
// reference scores are left absent.
func TestSimPOLossMatchesHandComputedValues(t *testing.T) {
	requirePreferenceLoss(t, PreferencePolicy{Mode: PreferenceSimPO, Scale: 2, Margin: 1}, PreferenceScores{
		PolicyChosen:   sequencescore.Score{LogProbability: -2, Tokens: 2},
		PolicyRejected: sequencescore.Score{LogProbability: -6, Tokens: 3},
	}, DPOResult{Loss: 0.31326168751822286, ChosenGradient: -0.2689414213699951, RejectedGradient: 0.17929428091333008})
}

// TestORPOLossMatchesHandComputedValues checks ORPO's loss (Hong, Lee and
// Thorne 2024): the chosen completion's mean token NLL plus
// λ·(-log σ(log(odds(yw|x)/odds(yl|x)))), where odds(y|x) = P/(1-P) and
// log P is the completion's mean token log probability. By hand, with a
// two-token chosen completion at P = 1/2 per token, a one-token rejected
// completion at P = 1/5 and λ = 0.1: the odds are 1 and 1/4, the log odds
// ratio is log 4 and σ(-log 4) = 1/5. The odds-ratio term is
// -log σ(log 4) = log(5/4) = 0.22314355131420976, and the loss is
// log 2 + 0.1·log(5/4) = 0.71546153569136628. The mean log probabilities'
// gradients are -1 - λ·(1/5)/(1 - 1/2) = -1.04 and λ·(1/5)/(1 - 1/5) =
// 0.025, so the summed ones' are -1.04/2 = -0.52 and 0.025/1 = 0.025. The
// loss reads no reference.
func TestORPOLossMatchesHandComputedValues(t *testing.T) {
	requirePreferenceLoss(t, PreferencePolicy{Mode: PreferenceORPO, Scale: 0.1}, PreferenceScores{
		PolicyChosen:   sequencescore.Score{LogProbability: 2 * math.Log(0.5), Tokens: 2},
		PolicyRejected: sequencescore.Score{LogProbability: math.Log(0.2), Tokens: 1},
	}, DPOResult{Loss: 0.71546153569136628, ChosenGradient: -0.52, RejectedGradient: 0.025})
}

// requirePreferenceLoss requires policy's loss on scores to match want.
func requirePreferenceLoss(t *testing.T, policy PreferencePolicy, scores PreferenceScores, want DPOResult) {
	t.Helper()
	got, err := policy.Loss(scores)
	if err != nil {
		t.Fatal(err)
	}
	const tolerance = 1e-12
	if math.Abs(got.Loss-want.Loss) > tolerance || math.Abs(got.ChosenGradient-want.ChosenGradient) > tolerance ||
		math.Abs(got.RejectedGradient-want.RejectedGradient) > tolerance {
		t.Fatalf("%s loss = %+v, want %+v", policy.Mode, got, want)
	}
}

// TestPreferenceModeGradientsFiniteDifference checks each mode's gradients
// with respect to the summed chosen and rejected log probabilities against
// central differences of its own loss, and DPO's own mode against DPOLoss.
func TestPreferenceModeGradientsFiniteDifference(t *testing.T) {
	scores := PreferenceScores{
		PolicyChosen:    sequencescore.Score{LogProbability: -3, Tokens: 2},
		PolicyRejected:  sequencescore.Score{LogProbability: -4, Tokens: 3},
		ReferenceChosen: sequencescore.Score{LogProbability: -2.5, Tokens: 2}, ReferenceRejected: sequencescore.Score{LogProbability: -4.5, Tokens: 3},
	}
	const epsilon = 1e-6
	for _, policy := range []PreferencePolicy{
		{Scale: 0.4}, {Mode: PreferenceSimPO, Scale: 2, Margin: 0.5}, {Mode: PreferenceORPO, Scale: 0.1},
	} {
		result, err := policy.Loss(scores)
		if err != nil {
			t.Fatal(err)
		}
		if dpo, _ := DPOLoss(scores, policy.Scale); policy.Mode == "" && dpo != result {
			t.Fatalf("DPO mode = %+v, DPOLoss = %+v", result, dpo)
		}
		for index, gradient := range []float64{result.ChosenGradient, result.RejectedGradient} {
			shifted := func(delta float64) float64 {
				moved := scores
				if index == 0 {
					moved.PolicyChosen.LogProbability += delta
				} else {
					moved.PolicyRejected.LogProbability += delta
				}
				value, err := policy.Loss(moved)
				if err != nil {
					t.Fatal(err)
				}
				return value.Loss
			}
			if want := (shifted(epsilon) - shifted(-epsilon)) / (2 * epsilon); math.Abs(gradient-want) > 1e-8 {
				t.Fatalf("%q gradient %d = %g, finite difference %g", policy.Mode, index, gradient, want)
			}
		}
	}
}

// TestPreferenceModesRejectInvalidPolicies refuses an undeclared mode, a
// margin outside SimPO or outside [0, ∞), class weights outside KTO or
// below zero, a nonpositive scale, an ORPO pair whose chosen completion the
// policy already predicts with certainty, a pair under KTO, and unpaired
// completions under a pair mode, alone, or without both classes to weigh.
func TestPreferenceModesRejectInvalidPolicies(t *testing.T) {
	scores := PreferenceScores{PolicyChosen: continuationScore(-1), PolicyRejected: continuationScore(-2)}
	for _, policy := range []PreferencePolicy{
		{Mode: "ipo", Scale: 0.1},
		{Scale: 0.1, Margin: 1},
		{Mode: PreferenceORPO, Scale: 0.1, Margin: 1},
		{Mode: PreferenceSimPO, Scale: 2, Margin: -1},
		{Mode: PreferenceSimPO, Scale: 2, Margin: math.NaN()},
		{Mode: PreferenceSimPO},
		{Scale: 0.1, DesirableWeight: 1, UndesirableWeight: 1},
		{Mode: PreferenceKTO, Scale: 0.1},
		{Mode: PreferenceKTO, Scale: 0.1, DesirableWeight: 1, UndesirableWeight: -1},
		{Mode: PreferenceKTO, Scale: 0.1, DesirableWeight: 1, UndesirableWeight: 1},
	} {
		if _, err := policy.Loss(scores); err == nil {
			t.Fatalf("accepted policy %+v", policy)
		}
	}
	scores.PolicyChosen = continuationScore(0)
	if _, err := (PreferencePolicy{Mode: PreferenceORPO, Scale: 0.1}).Loss(scores); err == nil {
		t.Fatal("ORPO accepted a chosen completion at probability one")
	}
	unpaired := UnpairedScores{
		Policy: continuationScore(-1), Reference: continuationScore(-1),
		MismatchedPolicy: continuationScore(-2), MismatchedReference: continuationScore(-2),
	}
	kto := PreferencePolicy{Mode: PreferenceKTO, Scale: 0.1, DesirableWeight: 1, UndesirableWeight: 1}
	for _, refused := range []struct {
		policy   PreferencePolicy
		examples []UnpairedScores
	}{
		{PreferencePolicy{Scale: 0.1}, []UnpairedScores{unpaired, unpaired}},
		{kto, []UnpairedScores{unpaired}},
		{kto, []UnpairedScores{unpaired, {}}},
	} {
		if _, err := refused.policy.UnpairedLoss(refused.examples); err == nil {
			t.Fatalf("%q accepted %d unpaired completions", refused.policy.Mode, len(refused.examples))
		}
	}
	if _, err := KTOPolicy(testutil.ArtifactID(t, artifact.KindModel, "reference"), 0.1, 3, 0); err == nil {
		t.Fatal("KTO weighed desirable completions against none")
	}
}
