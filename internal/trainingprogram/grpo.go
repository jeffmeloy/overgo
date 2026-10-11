package trainingprogram

import (
	"errors"

	"overgo/internal/artifact"
)

// rewardRule names the rule the reward objective weighs a group's answers by,
// and its version: by their order alone, each answer by the share of the
// others it beats less the share that beat it. A session plan for the
// objective carries it, so a change to the rule is another plan.
const rewardRule = "order/1"

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

// InstructionOrder is the one ordering of the answers to a prompt that the
// instruction objective trains by, as a reward: the more of the prompt's
// rules an answer passes the higher, and among answers that pass every rule
// the fewer tokens it generated the higher. Answers that fail the same
// number of rules stand together whatever their length: stopping early earns
// nothing short of passing. Only the order of its values means anything.
func InstructionOrder(passed, rules, tokens uint64) float64 {
	if passed < rules {
		return float64(passed)
	}
	return float64(rules) + 1/float64(tokens+1)
}

// OrderGradients weighs the answers of one group by the order of their
// rewards alone. An answer's weight is the share of the group's other answers
// it beats less the share that beat it, an equal reward counting for neither,
// so the weights sum to nothing, and any rewards in the same order give the
// same weights. What returns is, for each answer, the gradient of the group's
// loss with respect to the logarithm of the answer's probability: the weight
// with its sign turned, since the trainer descends and a reward is climbed,
// over the group's size. A group whose answers all stand together returns all
// zeros: it has nothing to teach, and is to move nothing.
func OrderGradients(rewards []float64) ([]float64, error) {
	if len(rewards) < 2 {
		return nil, errors.New("training program: a reward group holds at least two answers")
	}
	gradients := make([]float64, len(rewards))
	for index, reward := range rewards {
		if !finite(reward) {
			return nil, errors.New("training program: a reward is not a finite number")
		}
		// Counted with the sign already turned: a loss to another answer adds, a win takes away.
		var standing float64
		for _, other := range rewards {
			switch {
			case reward < other:
				standing++
			case reward > other:
				standing--
			}
		}
		gradients[index] = standing / float64(len(rewards)-1) / float64(len(rewards))
	}
	return gradients, nil
}
