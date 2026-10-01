package trainingdata

import (
	"errors"
	"fmt"
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

// UnpairedExample is one unpaired completion KTO trains on: its prompt and
// completion as one sequence, its label, and the mismatched sequence its
// KL estimate scores, this example's prompt with another's completion.
type UnpairedExample struct {
	ID, Group  string
	Desirable  bool
	Sequence   PreferenceSequence
	Mismatched PreferenceSequence
}

// UnpairedBatch is one KTO microbatch and the stream state after it.
type UnpairedBatch struct {
	Examples []UnpairedExample
	State    StreamState
}

// CompileUnpairedBatch decodes a microbatch of unpaired completions, each
// one value whose role labels it desirable (chosen) or undesirable
// (rejected) and whose completion mask selects a suffix after its prompt.
// Each example's mismatched sequence is its prompt with the next example's
// completion, cyclically, so the microbatch must hold two or more distinct
// records for every mismatched pair to cross examples.
func CompileUnpairedBatch(batch Batch, decode TokenDecoder) (UnpairedBatch, error) {
	if len(batch.Examples) <= 1 || !batch.State.Identity.Valid() || decode == nil {
		return UnpairedBatch{}, errors.New("training data: an unpaired microbatch needs two or more examples and a decoder")
	}
	result := UnpairedBatch{Examples: make([]UnpairedExample, len(batch.Examples)), State: batch.State}
	seen := make(map[string]bool, len(batch.Examples))
	for index, example := range batch.Examples {
		if example.ID == "" || seen[example.ID] || len(example.Values) != 1 ||
			example.Values[0].Role != RoleChosen && example.Values[0].Role != RoleRejected {
			return UnpairedBatch{}, errors.New("training data: an unpaired example is one labelled value of a record its microbatch holds once")
		}
		seen[example.ID] = true
		value := example.Values[0]
		tokens, err := decode(value)
		if err != nil {
			return UnpairedBatch{}, err
		}
		if completionStart(tokens, value.Completion) < 0 {
			return UnpairedBatch{}, fmt.Errorf("training data: unpaired example %q is not a prompt followed by its completion", example.ID)
		}
		result.Examples[index] = UnpairedExample{
			ID: example.ID, Group: example.Group, Desirable: value.Role == RoleChosen,
			Sequence: PreferenceSequence{Tokens: slices.Clone(tokens), Completion: slices.Clone(value.Completion)},
		}
	}
	for index := range result.Examples {
		next := result.Examples[(index+1)%len(result.Examples)]
		var err error
		if result.Examples[index].Mismatched, err = SpliceCompletion(result.Examples[index].Sequence, next.Sequence); err != nil {
			return UnpairedBatch{}, err
		}
	}
	return result, nil
}

// SpliceCompletion is prompt's tokens before its completion followed by
// completion's completion tokens: the mismatched sequence KTO's KL
// estimate scores. Both sequences must select a suffix after their prompt.
func SpliceCompletion(prompt, completion PreferenceSequence) (PreferenceSequence, error) {
	promptEnd := completionStart(prompt.Tokens, prompt.Completion)
	start := completionStart(completion.Tokens, completion.Completion)
	if promptEnd < 0 || start < 0 {
		return PreferenceSequence{}, errors.New("training data: a spliced sequence is a prompt followed by its completion")
	}
	tokens := append(slices.Clone(prompt.Tokens[:promptEnd]), completion.Tokens[start:]...)
	mask := make([]bool, len(tokens))
	for index := promptEnd; index < len(mask); index++ {
		mask[index] = true
	}
	return PreferenceSequence{Tokens: tokens, Completion: mask}, nil
}

// completionStart is where a valid sequence's completion begins when its
// mask selects exactly the suffix from there, else -1.
func completionStart(tokens []int, completion []bool) int {
	if !validPreferenceSequence(tokens, completion) {
		return -1
	}
	start := slices.Index(completion, true)
	if slices.Contains(completion[start:], false) {
		return -1
	}
	return start
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
