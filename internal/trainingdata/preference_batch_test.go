package trainingdata

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestDPOPreferenceBatchCompilesExactPrefixAndMasks(t *testing.T) {
	example := Example{ID: "pair", Group: "group", Values: []Value{
		{Role: RoleChosen, Data: []byte{1, 2, 3}, Completion: []bool{false, false, true}},
		{Role: RoleRejected, Data: []byte{1, 2, 4}, Completion: []bool{false, false, true}},
	}}
	identity := testutil.ArtifactID(t, artifact.KindProfile, "preference-stream")
	batch, err := CompilePreferenceBatch(Batch{Examples: []Example{example}, State: StreamState{Identity: identity, Position: 1}}, func(value Value) ([]int, error) {
		result := make([]int, len(value.Data))
		for index, token := range value.Data {
			result[index] = int(token)
		}
		return result, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Pairs) != 1 || batch.State.Identity != identity || batch.Pairs[0].SharedPrefix != 2 ||
		!batch.Pairs[0].Chosen.Completion[2] || !batch.Pairs[0].Rejected.Completion[2] {
		t.Fatalf("compiled batch=%+v", batch)
	}
}

// TestUnpairedBatchSplicesEachPromptWithTheNextCompletion compiles three
// labelled completions and requires each example's mismatched sequence to
// be its own prompt followed by the next example's completion, cyclically,
// masked from the prompt's end. A microbatch of one, a record held twice,
// a value that is not a labelled completion, and a mask that is no suffix
// are refused.
func TestUnpairedBatchSplicesEachPromptWithTheNextCompletion(t *testing.T) {
	identity := testutil.ArtifactID(t, artifact.KindProfile, "unpaired-stream")
	value := func(role ValueRole, tokens []int, prompt int) Value {
		mask := make([]bool, len(tokens))
		for index := prompt; index < len(mask); index++ {
			mask[index] = true
		}
		result, err := TokenValue(role, tokens, mask)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	examples := []Example{
		{ID: "a", Values: []Value{value(RoleChosen, []int{1, 2, 7}, 2)}},
		{ID: "b", Values: []Value{value(RoleRejected, []int{3, 8, 9}, 1)}},
		{ID: "c", Values: []Value{value(RoleChosen, []int{4, 5, 6, 10}, 3)}},
	}
	batch, err := CompileUnpairedBatch(Batch{Examples: examples, State: StreamState{Identity: identity, Position: 3}}, TokenIDs)
	if err != nil {
		t.Fatal(err)
	}
	want := []PreferenceSequence{
		{Tokens: []int{1, 2, 8, 9}, Completion: []bool{false, false, true, true}},
		{Tokens: []int{3, 10}, Completion: []bool{false, true}},
		{Tokens: []int{4, 5, 6, 7}, Completion: []bool{false, false, false, true}},
	}
	for index, example := range batch.Examples {
		if example.Desirable != (index != 1) || !slices.Equal(example.Mismatched.Tokens, want[index].Tokens) ||
			!slices.Equal(example.Mismatched.Completion, want[index].Completion) {
			t.Fatalf("example %s = %+v, want desirable %t and mismatched %+v", example.ID, example, index != 1, want[index])
		}
	}
	interleaved := value(RoleChosen, []int{1, 2, 3}, 1)
	interleaved.Completion = []bool{false, true, false}
	for name, refused := range map[string][]Example{
		"one example":         examples[:1],
		"a record held twice": {examples[0], examples[0]},
		"an input value":      {examples[0], {ID: "d", Values: []Value{{Role: RoleInput, Data: []byte("x")}}}},
		"a mask no suffix":    {examples[0], {ID: "e", Values: []Value{interleaved}}},
	} {
		if _, err := CompileUnpairedBatch(Batch{Examples: refused, State: StreamState{Identity: identity, Position: 2}}, TokenIDs); err == nil {
			t.Fatalf("compiled %s", name)
		}
	}
}
