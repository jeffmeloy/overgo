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

type DPOObservation struct {
	Step              uint64  `json:"step"`
	Loss              float64 `json:"loss"`
	PolicyChosen      float64 `json:"policy_chosen"`
	PolicyRejected    float64 `json:"policy_rejected"`
	ReferenceChosen   float64 `json:"reference_chosen"`
	ReferenceRejected float64 `json:"reference_rejected"`
	PolicyMargin      float64 `json:"policy_margin"`
	ReferenceMargin   float64 `json:"reference_margin"`
	RelativeMargin    float64 `json:"relative_margin"`
	ChosenTokens      uint64  `json:"chosen_tokens"`
	RejectedTokens    uint64  `json:"rejected_tokens"`
	LearningRate      float64 `json:"learning_rate"`
	GradientL2        float64 `json:"gradient_l2"`
	UpdateL2          float64 `json:"update_l2"`
}

func (observation DPOObservation) Valid() bool {
	values := [...]float64{
		observation.Loss, observation.PolicyChosen, observation.PolicyRejected,
		observation.ReferenceChosen, observation.ReferenceRejected,
		observation.PolicyMargin, observation.ReferenceMargin, observation.RelativeMargin,
		observation.LearningRate, observation.GradientL2, observation.UpdateL2,
	}
	if observation.Step == 0 || observation.ChosenTokens == 0 || observation.RejectedTokens == 0 ||
		observation.Loss < 0 || observation.LearningRate <= 0 || observation.GradientL2 < 0 || observation.UpdateL2 < 0 {
		return false
	}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return observation.PolicyMargin == observation.PolicyChosen-observation.PolicyRejected &&
		observation.ReferenceMargin == observation.ReferenceChosen-observation.ReferenceRejected &&
		observation.RelativeMargin == observation.PolicyMargin-observation.ReferenceMargin
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
