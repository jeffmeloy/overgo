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

func CompilePreferenceBatch(examples []Example, decode TokenDecoder) ([]PreferencePair, error) {
	if len(examples) == 0 || decode == nil {
		return nil, errors.New("training data: preference examples or decoder absent")
	}
	result := make([]PreferencePair, len(examples))
	for index, example := range examples {
		chosen, rejected, err := preferenceValues(example)
		if err != nil {
			return nil, err
		}
		chosenTokens, err := decode(chosen)
		if err != nil {
			return nil, err
		}
		rejectedTokens, err := decode(rejected)
		if err != nil {
			return nil, err
		}
		if !validPreferenceSequence(chosenTokens, chosen.Completion) ||
			!validPreferenceSequence(rejectedTokens, rejected.Completion) {
			return nil, errors.New("training data: invalid preference sequence")
		}
		prefix := commonTokenPrefix(chosenTokens, rejectedTokens)
		if prefix == 0 {
			return nil, errors.New("training data: preference pair has no exact prefix")
		}
		result[index] = PreferencePair{
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
