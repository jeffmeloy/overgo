package trainingdata

import (
	"errors"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/recipecontract"
)

const EncodingTokenIDs = "token-ids-u64-le"

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
		prefix := CommonTokenPrefix(chosenTokens, rejectedTokens)
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

func CommonTokenPrefix(left, right []int) int {
	limit := min(len(left), len(right))
	for index := range limit {
		if left[index] != right[index] {
			return index
		}
	}
	return limit
}

func TokenValue(role ValueRole, tokens []int, completion []bool) (Value, error) {
	bytes, ok := checked.Mul64(uint64(len(tokens)), binaryschema.Uint64Bytes)
	size, fits := checked.Int(bytes)
	if !ok || !fits || !validPreferenceSequence(tokens, completion) || role != RoleChosen && role != RoleRejected {
		return Value{}, errors.New("training data: invalid preference token value")
	}
	data := make([]byte, size)
	for index, token := range tokens {
		if token < 0 {
			return Value{}, errors.New("training data: negative token ID")
		}
		binaryschema.LittleEndian.PutUint64(data[index*binaryschema.Uint64Bytes:], uint64(token))
	}
	return Value{
		Role: role, Modality: recipecontract.ModalityText, Encoding: EncodingTokenIDs,
		Shape: []int{len(tokens)}, Data: data, Completion: slices.Clone(completion),
	}, nil
}

func PreferenceTokenValues(chosenTokens, rejectedTokens []int) (Value, Value, error) {
	prefix := CommonTokenPrefix(chosenTokens, rejectedTokens)
	if prefix == 0 || prefix >= len(chosenTokens) || prefix >= len(rejectedTokens) {
		return Value{}, Value{}, errors.New("training data: preference tokens require shared prefix and distinct suffixes")
	}
	mask := func(length int) []bool {
		result := make([]bool, length)
		for index := prefix; index < length; index++ {
			result[index] = true
		}
		return result
	}
	chosen, err := TokenValue(RoleChosen, chosenTokens, mask(len(chosenTokens)))
	if err != nil {
		return Value{}, Value{}, err
	}
	rejected, err := TokenValue(RoleRejected, rejectedTokens, mask(len(rejectedTokens)))
	return chosen, rejected, err
}

func TokenIDs(value Value) ([]int, error) {
	if value.Encoding != EncodingTokenIDs || len(value.Data) == 0 || len(value.Data)%binaryschema.Uint64Bytes != 0 {
		return nil, errors.New("training data: invalid token ID payload")
	}
	result := make([]int, len(value.Data)/binaryschema.Uint64Bytes)
	for index := range result {
		encoded := binaryschema.LittleEndian.Uint64(value.Data[index*binaryschema.Uint64Bytes:])
		var ok bool
		if result[index], ok = checked.Int(encoded); !ok {
			return nil, errors.New("training data: token ID exceeds host integer")
		}
	}
	return result, nil
}
