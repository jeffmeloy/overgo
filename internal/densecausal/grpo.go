package densecausal

import (
	"errors"
	"math"
	"slices"

	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// TrainGRPOGroupsResume trains on reward groups, one update a group whose
// answers' rewards put them in some order (trainingprogram.OrderGradients).
// A group whose answers all stand together takes no update: the weights, the
// momentum and the optimizer's step stay exactly what they were, the stream's
// position passes the group, and the state returned counts it as skipped. It
// is observed by nothing, since nothing happened to observe.
func (m *Model) TrainGRPOGroupsResume(
	groups []trainingdata.RolloutGroup,
	baseLR, momentum float64,
	resume *RLState,
	observe func(trainingprogram.GRPOObservation),
) ([]trainingprogram.GRPOObservation, RLState, error) {
	if len(groups) == 0 {
		return nil, RLState{}, errors.New("densecausal: GRPO groups absent")
	}
	run, err := m.newRLRun("GRPO", baseLR, momentum, resume)
	if err != nil {
		return nil, RLState{}, err
	}
	observations := make([]trainingprogram.GRPOObservation, 0)
	for _, group := range groups {
		if err := run.admit(group.State); err != nil {
			return nil, RLState{}, err
		}
		observation, grads, err := m.grpoLossAndGrads(group)
		if err != nil {
			return nil, RLState{}, err
		}
		if grads == nil {
			run.skipped++
			continue
		}
		observation.Step, observation.LearningRate, observation.GradientL2, observation.UpdateL2, err = run.stepObserved(grads)
		if err != nil {
			return nil, RLState{}, err
		}
		if observe != nil {
			observe(observation)
		} else {
			observations = append(observations, observation)
		}
	}
	return observations, run.state(), nil
}

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

// grpoLossAndGrads is a reward group's loss and its gradients, or no
// gradients at all for a group whose answers all stand together, which is
// found from the rewards before any answer is scored.
func (m *Model) grpoLossAndGrads(group trainingdata.RolloutGroup) (trainingprogram.GRPOObservation, Grads, error) {
	rewards := make([]float64, len(group.Rollouts))
	for index, rollout := range group.Rollouts {
		rewards[index] = rollout.Reward
	}
	gradients, err := trainingprogram.OrderGradients(rewards)
	if err != nil || !slices.ContainsFunc(gradients, func(gradient float64) bool { return gradient != 0 }) {
		return trainingprogram.GRPOObservation{}, nil, err
	}
	observation := RewardObservation(group)
	traces := make([]preferenceTrace, len(group.Rollouts))
	for index, rollout := range group.Rollouts {
		trace, err := m.sequenceTrace(rollout.Sequence)
		if err != nil {
			return trainingprogram.GRPOObservation{}, nil, err
		}
		traces[index] = trace
		observation.Loss += gradients[index] * trace.score.LogProbability
	}
	grads, err := m.sequenceGradientSum(traces, gradients)
	return observation, grads, err
}
