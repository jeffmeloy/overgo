package densecausal

import (
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// WeightedSequence is one sequence of a weighted token-prediction step: its
// tokens and, for each token after the first, the weight of predicting it
// from those before. The step's loss is the weights' sum of the tokens'
// next-token losses and nothing else: a weight may be negative, to make a
// token less likely, or zero, to leave it out, and no mean is taken over the
// tokens, so a sequence's length weighs only as its weights say. A sequence
// with no weights is weighed by the mean, one over the count of its predicted
// tokens for each, which is a fine-tuning step.
type WeightedSequence struct {
	Tokens  []int
	Weights []float32
}

// RewardSequences is a reward group as the weighted step trains on it. Each
// answer is one sequence; every token of its completion carries the answer's
// weight by the order of the group's rewards (trainingprogram.OrderGradients),
// with the sign a next-token loss takes: an answer that beats the others has
// its completion's loss weighed positively, so descending makes it more
// likely, and one that is beaten negatively. Prompt tokens carry no weight,
// and neither does anything of a group whose answers all stand together, which
// the step then passes without an update. The weight is the same for every
// token of a completion and is not divided by their count: an answer's weight
// is its standing, whatever its length.
func RewardSequences(group trainingdata.RolloutGroup) ([]WeightedSequence, error) {
	rewards := make([]float64, len(group.Rollouts))
	for index, rollout := range group.Rollouts {
		rewards[index] = rollout.Reward
	}
	gradients, err := trainingprogram.OrderGradients(rewards)
	if err != nil {
		return nil, err
	}
	sequences := make([]WeightedSequence, len(group.Rollouts))
	for index, rollout := range group.Rollouts {
		weights := make([]float32, len(rollout.Sequence.Tokens)-1)
		for token := range weights {
			if rollout.Sequence.Completion[token+1] {
				weights[token] = float32(0 - gradients[index])
			}
		}
		sequences[index] = WeightedSequence{Tokens: rollout.Sequence.Tokens, Weights: weights}
	}
	return sequences, nil
}
