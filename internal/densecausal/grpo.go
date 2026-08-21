package densecausal

import (
	"errors"

	"overgo/internal/optimizer"
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
	if resume != nil && !resume.Stream.Identity.Valid() {
		return nil, RLState{}, errors.New("densecausal: GRPO resume stream authority absent")
	}
	names, weights, gradients, plan, err := m.trainSetup()
	if err != nil {
		return nil, RLState{}, err
	}
	if baseLR <= 0 {
		return nil, RLState{}, errors.New("densecausal: positive recipe-derived learning rate required")
	}
	update, err := optimizer.New(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: baseLR, Momentum: momentum, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, RLState{}, err
	}
	if resume != nil {
		if err := update.Restore(resume.Optimizer); err != nil {
			return nil, RLState{}, err
		}
	}
	stream := trainingdata.StreamState{}
	if resume != nil {
		stream = resume.Stream
	}
	observations := make([]trainingprogram.GRPOObservation, 0)
	for _, group := range groups {
		if !group.State.Identity.Valid() || stream.Identity.Valid() &&
			(group.State.Identity != stream.Identity || group.State.Position <= stream.Position) {
			return nil, RLState{}, errors.New("densecausal: GRPO stream authority differs")
		}
		scatter(m, names, weights)
		observation, grads, err := m.grpoLossAndGrads(group, scale)
		if err != nil {
			return nil, RLState{}, err
		}
		m.gatherGrads(names, gradients, grads)
		step := update.Step()
		observation.Step = uint64(step.Step)
		observation.LearningRate = step.LearningRate
		observation.GradientL2 = step.GradientL2
		observation.UpdateL2 = step.UpdateL2
		if observe != nil {
			observe(observation)
		} else {
			observations = append(observations, observation)
		}
		stream = group.State
	}
	scatter(m, names, weights)
	return observations, RLState{Optimizer: update.Snapshot(), Stream: stream}, nil
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
