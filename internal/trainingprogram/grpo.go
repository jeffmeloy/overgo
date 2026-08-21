package trainingprogram

import (
	"errors"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
)

type GroupedScore struct {
	Score  sequencescore.Score
	Reward float64
}

type GRPOResult struct {
	Loss             float64
	ScoreGradients   []float64
	MeanReward       float64
	RewardDispersion float64
}

type GRPOObservation struct {
	Step             uint64      `json:"step"`
	Loss             float64     `json:"loss"`
	MeanReward       float64     `json:"mean_reward"`
	RewardDispersion float64     `json:"reward_dispersion"`
	GroupSize        uint64      `json:"group_size"`
	CompletionTokens uint64      `json:"completion_tokens"`
	Evaluator        artifact.ID `json:"evaluator"`
	LearningRate     float64     `json:"learning_rate"`
	GradientL2       float64     `json:"gradient_l2"`
	UpdateL2         float64     `json:"update_l2"`
}

func (observation GRPOObservation) Valid() bool {
	values := [...]float64{observation.Loss, observation.MeanReward, observation.RewardDispersion,
		observation.LearningRate, observation.GradientL2, observation.UpdateL2}
	if observation.Step == 0 || observation.GroupSize < 2 || observation.CompletionTokens < observation.GroupSize ||
		observation.Evaluator.Kind() != artifact.KindEvidence || observation.RewardDispersion < 0 ||
		observation.LearningRate <= 0 || observation.GradientL2 < 0 || observation.UpdateL2 < 0 {
		return false
	}
	for _, value := range values {
		if !finite(value) {
			return false
		}
	}
	return true
}

// GRPOLoss: centered, RMS-normalized group reward objective.
func GRPOLoss(group []GroupedScore, scale float64) (GRPOResult, error) {
	if len(group) < 2 || scale <= 0 || !finite(scale) {
		return GRPOResult{}, errors.New("training program: invalid GRPO group or scale")
	}
	var mean float64
	for _, item := range group {
		if !item.Score.Valid() || !finite(item.Reward) {
			return GRPOResult{}, errors.New("training program: invalid GRPO score or reward")
		}
		mean += item.Reward
	}
	mean /= float64(len(group))
	var squareSum float64
	for _, item := range group {
		centered := item.Reward - mean
		squareSum += centered * centered
	}
	dispersion := math.Sqrt(squareSum / float64(len(group)))
	result := GRPOResult{ScoreGradients: make([]float64, len(group)), MeanReward: mean, RewardDispersion: dispersion}
	if dispersion == 0 {
		return result, nil
	}
	denominator := dispersion * float64(len(group))
	for index, item := range group {
		gradient := -scale * (item.Reward - mean) / denominator
		result.ScoreGradients[index] = gradient
		result.Loss += gradient * item.Score.LogProbability
	}
	return result, nil
}
