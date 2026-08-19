package trainingprogram

import (
	"math"
	"testing"

	"overgo/internal/sequencescore"
)

func TestGRPORelativeRewardsOwnScoreGradients(t *testing.T) {
	group := []GroupedScore{
		{Score: sequencescore.Score{LogProbability: -3, Tokens: 1}, Reward: -1},
		{Score: sequencescore.Score{LogProbability: -2, Tokens: 1}, Reward: 0},
		{Score: sequencescore.Score{LogProbability: -1, Tokens: 1}, Reward: 1},
	}
	result, err := GRPOLoss(group, 1)
	if err != nil {
		t.Fatal(err)
	}
	var gradientSum float64
	for _, gradient := range result.ScoreGradients {
		gradientSum += gradient
	}
	if math.Abs(gradientSum) > 1e-15 || result.ScoreGradients[0] <= 0 || result.ScoreGradients[2] >= 0 || result.MeanReward != 0 {
		t.Fatalf("result=%+v", result)
	}
}

func TestGRPOEqualRewardsProduceNoUpdate(t *testing.T) {
	group := []GroupedScore{
		{Score: sequencescore.Score{LogProbability: -2, Tokens: 1}, Reward: 1},
		{Score: sequencescore.Score{LogProbability: -1, Tokens: 1}, Reward: 1},
	}
	result, err := GRPOLoss(group, 1)
	if err != nil {
		t.Fatal(err)
	}
	if result.Loss != 0 || result.RewardDispersion != 0 || result.ScoreGradients[0] != 0 || result.ScoreGradients[1] != 0 {
		t.Fatalf("result=%+v", result)
	}
}
