package trainingdata

import (
	"errors"
	"slices"
)

type TokenDecoder func(Value) ([]int, error)

type PreferenceSequence struct {
	Tokens     []int
	Completion []bool
}

type PreferencePair struct {
	ID           string
	Group        string
	SharedPrefix int
	Chosen       PreferenceSequence
	Rejected     PreferenceSequence
}

type PreferenceBatch struct {
	Pairs []PreferencePair
	State StreamState
}

func CompilePreferenceBatch(batch Batch, decode TokenDecoder) (PreferenceBatch, error) {
	if len(batch.Examples) == 0 || !batch.State.Identity.Valid() || decode == nil {
		return PreferenceBatch{}, errors.New("training data: preference examples or decoder absent")
	}
	result := PreferenceBatch{Pairs: make([]PreferencePair, len(batch.Examples)), State: batch.State}
	for index, example := range batch.Examples {
		chosen, rejected, err := preferenceValues(example)
		if err != nil {
			return PreferenceBatch{}, err
		}
		chosenTokens, err := decode(chosen)
		if err != nil {
			return PreferenceBatch{}, err
		}
		rejectedTokens, err := decode(rejected)
		if err != nil {
			return PreferenceBatch{}, err
		}
		if !validPreferenceSequence(chosenTokens, chosen.Completion) ||
			!validPreferenceSequence(rejectedTokens, rejected.Completion) {
			return PreferenceBatch{}, errors.New("training data: invalid preference sequence")
		}
		prefix := commonTokenPrefix(chosenTokens, rejectedTokens)
		if prefix == 0 {
			return PreferenceBatch{}, errors.New("training data: preference pair has no exact prefix")
		}
		result.Pairs[index] = PreferencePair{
			ID: example.ID, Group: example.Group, SharedPrefix: prefix,
			Chosen:   PreferenceSequence{Tokens: slices.Clone(chosenTokens), Completion: slices.Clone(chosen.Completion)},
			Rejected: PreferenceSequence{Tokens: slices.Clone(rejectedTokens), Completion: slices.Clone(rejected.Completion)},
		}
	}
	return result, nil
}

func preferenceValues(example Example) (Value, Value, error) {
	var chosen, rejected Value
	var chosenCount, rejectedCount int
	for _, value := range example.Values {
		switch value.Role {
		case RoleChosen:
			chosen, chosenCount = value, chosenCount+1
		case RoleRejected:
			rejected, rejectedCount = value, rejectedCount+1
		}
	}
	if example.ID == "" || chosenCount != 1 || rejectedCount != 1 {
		return Value{}, Value{}, errors.New("training data: preference pair requires one chosen and one rejected value")
	}
	return chosen, rejected, nil
}

func validPreferenceSequence(tokens []int, completion []bool) bool {
	if len(tokens) < 2 || len(tokens) != len(completion) || completion[0] {
		return false
	}
	for _, selected := range completion[1:] {
		if selected {
			return true
		}
	}
	return false
}

func commonTokenPrefix(left, right []int) int {
	limit := min(len(left), len(right))
	for index := range limit {
		if left[index] != right[index] {
			return index
		}
	}
	return limit
}
