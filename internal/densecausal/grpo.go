package densecausal

import (
	"math"

	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// RewardObservation is what a reward group shows before it is stepped: its
// size, its evaluator, the tokens of its answers' completions, and the mean
// and the dispersion of its rewards as they were given. The loss reads the
// rewards' order and not their sizes; these describe the group to whoever
// watches the run.
func RewardObservation(group trainingdata.RolloutGroup) trainingprogram.GRPOObservation {
	observation := trainingprogram.GRPOObservation{GroupSize: uint64(len(group.Rollouts)), Evaluator: group.Evaluator}
	count := float64(len(group.Rollouts))
	for _, rollout := range group.Rollouts {
		observation.MeanReward += rollout.Reward / count
		for _, completed := range rollout.Sequence.Completion {
			if completed {
				observation.CompletionTokens++
			}
		}
	}
	for _, rollout := range group.Rollouts {
		observation.RewardDispersion += (rollout.Reward - observation.MeanReward) * (rollout.Reward - observation.MeanReward) / count
	}
	observation.RewardDispersion = math.Sqrt(observation.RewardDispersion)
	return observation
}
