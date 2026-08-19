package trainingdata

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/artifact"
)

type Rollout struct {
	ID       string
	Sequence PreferenceSequence
	Reward   float64
}

type RolloutGroup struct {
	ID        string
	Evaluator artifact.ID
	Rollouts  []Rollout
	State     StreamState
}

func NewRolloutGroup(id string, evaluator artifact.ID, state StreamState, rollouts []Rollout) (RolloutGroup, error) {
	if id == "" || evaluator.Kind() != artifact.KindEvidence || !state.Identity.Valid() || len(rollouts) < 2 {
		return RolloutGroup{}, errors.New("training data: invalid rollout group authority")
	}
	result := RolloutGroup{ID: id, Evaluator: evaluator, State: state, Rollouts: make([]Rollout, len(rollouts))}
	seen := make(map[string]struct{}, len(rollouts))
	for index, rollout := range rollouts {
		if rollout.ID == "" || math.IsNaN(rollout.Reward) || math.IsInf(rollout.Reward, 0) {
			return RolloutGroup{}, errors.New("training data: invalid grouped rollout identity or reward")
		}
		if !validPreferenceSequence(rollout.Sequence.Tokens, rollout.Sequence.Completion) {
			return RolloutGroup{}, fmt.Errorf("training data: invalid grouped rollout sequence %q (%d tokens, %d mask)",
				rollout.ID, len(rollout.Sequence.Tokens), len(rollout.Sequence.Completion))
		}
		if _, exists := seen[rollout.ID]; exists {
			return RolloutGroup{}, errors.New("training data: duplicate rollout identity")
		}
		seen[rollout.ID] = struct{}{}
		result.Rollouts[index] = Rollout{ID: rollout.ID, Reward: rollout.Reward, Sequence: PreferenceSequence{
			Tokens: slices.Clone(rollout.Sequence.Tokens), Completion: slices.Clone(rollout.Sequence.Completion),
		}}
	}
	return result, nil
}
