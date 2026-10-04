package densecausal

import (
	"errors"

	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

func (m *Model) TrainGRPOGroupsResume(
	groups []trainingdata.RolloutGroup,
	baseLR, momentum, scale float64,
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
		observation, grads, err := m.grpoLossAndGrads(group, scale)
		if err != nil {
			return nil, RLState{}, err
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

func (m *Model) grpoLossAndGrads(group trainingdata.RolloutGroup, scale float64) (trainingprogram.GRPOObservation, Grads, error) {
	traces := make([]preferenceTrace, len(group.Rollouts))
	scores := make([]trainingprogram.GroupedScore, len(group.Rollouts))
	var completionTokens uint64
	for index, rollout := range group.Rollouts {
		trace, err := m.sequenceTrace(rollout.Sequence)
		if err != nil {
			return trainingprogram.GRPOObservation{}, nil, err
		}
		traces[index] = trace
		scores[index] = trainingprogram.GroupedScore{Score: trace.score, Reward: rollout.Reward}
		completionTokens += trace.score.Tokens
	}
	objective, err := trainingprogram.GRPOLoss(scores, scale)
	if err != nil {
		return trainingprogram.GRPOObservation{}, nil, err
	}
	grads, err := m.sequenceGradientSum(traces, objective.ScoreGradients)
	if err != nil {
		return trainingprogram.GRPOObservation{}, nil, err
	}
	return trainingprogram.GRPOObservation{
		Loss: objective.Loss, MeanReward: objective.MeanReward, RewardDispersion: objective.RewardDispersion,
		GroupSize: uint64(len(group.Rollouts)), CompletionTokens: completionTokens, Evaluator: group.Evaluator,
	}, grads, nil
}
