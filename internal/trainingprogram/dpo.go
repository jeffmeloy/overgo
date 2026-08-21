package trainingprogram

import (
	"errors"
	"math"

	"overgo/internal/sequencescore"
)

// PreferenceScores are sequence log probabilities for one preference pair.
type PreferenceScores struct {
	PolicyChosen      sequencescore.Score
	PolicyRejected    sequencescore.Score
	ReferenceChosen   sequencescore.Score
	ReferenceRejected sequencescore.Score
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
		if !finite(value) {
			return false
		}
	}
	return observation.PolicyMargin == observation.PolicyChosen-observation.PolicyRejected &&
		observation.ReferenceMargin == observation.ReferenceChosen-observation.ReferenceRejected &&
		observation.RelativeMargin == observation.PolicyMargin-observation.ReferenceMargin
}

func DPOLoss(scores PreferenceScores, scale float64) (DPOResult, error) {
	if scale <= 0 || !finite(scale) ||
		!scores.PolicyChosen.Valid() || !scores.PolicyRejected.Valid() ||
		!scores.ReferenceChosen.Valid() || !scores.ReferenceRejected.Valid() {
		return DPOResult{}, errors.New("training program: invalid DPO scores or scale")
	}
	policy, err := sequencescore.Select(
		[]sequencescore.Score{scores.PolicyChosen, scores.PolicyRejected}, sequencescore.NormalizationSum,
	)
	if err != nil {
		return DPOResult{}, err
	}
	reference, err := sequencescore.Select(
		[]sequencescore.Score{scores.ReferenceChosen, scores.ReferenceRejected}, sequencescore.NormalizationSum,
	)
	if err != nil {
		return DPOResult{}, err
	}
	margin := scale * ((policy.Values[0] - policy.Values[1]) - (reference.Values[0] - reference.Values[1]))
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
