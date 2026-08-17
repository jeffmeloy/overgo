package trainingprogram

import (
	"errors"
	"math"
)

// PreferenceScores are sequence log probabilities for one preference pair.
type PreferenceScores struct {
	PolicyChosen      float64
	PolicyRejected    float64
	ReferenceChosen   float64
	ReferenceRejected float64
}

// DPOResult carries loss and policy score gradients. Reference scores are frozen.
type DPOResult struct {
	Loss             float64
	ChosenGradient   float64
	RejectedGradient float64
}

func DPOLoss(scores PreferenceScores, scale float64) (DPOResult, error) {
	if scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) ||
		math.IsNaN(scores.PolicyChosen) || math.IsInf(scores.PolicyChosen, 0) ||
		math.IsNaN(scores.PolicyRejected) || math.IsInf(scores.PolicyRejected, 0) ||
		math.IsNaN(scores.ReferenceChosen) || math.IsInf(scores.ReferenceChosen, 0) ||
		math.IsNaN(scores.ReferenceRejected) || math.IsInf(scores.ReferenceRejected, 0) {
		return DPOResult{}, errors.New("training program: invalid DPO scores or scale")
	}
	margin := scale * ((scores.PolicyChosen - scores.PolicyRejected) -
		(scores.ReferenceChosen - scores.ReferenceRejected))
	loss := softplus(-margin)
	down := sigmoid(-margin)
	return DPOResult{
		Loss: loss, ChosenGradient: -scale * down, RejectedGradient: scale * down,
	}, nil
}

func softplus(value float64) float64 {
	if value > 0 {
		return value + math.Log1p(math.Exp(-value))
	}
	return math.Log1p(math.Exp(value))
}

func sigmoid(value float64) float64 {
	if value >= 0 {
		inverse := math.Exp(-value)
		return 1 / (1 + inverse)
	}
	exponential := math.Exp(value)
	return exponential / (1 + exponential)
}
