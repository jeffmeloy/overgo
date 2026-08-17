package trainingdata

import "testing"

func TestDPOPreferenceBatchCompilesExactPrefixAndMasks(t *testing.T) {
	example := Example{ID: "pair", Group: "group", Values: []Value{
		{Role: RoleChosen, Data: []byte{1, 2, 3}, Completion: []bool{false, false, true}},
		{Role: RoleRejected, Data: []byte{1, 2, 4}, Completion: []bool{false, false, true}},
	}}
	pairs, err := CompilePreferenceBatch([]Example{example}, func(value Value) ([]int, error) {
		result := make([]int, len(value.Data))
		for index, token := range value.Data {
			result[index] = int(token)
		}
		return result, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 1 || pairs[0].SharedPrefix != 2 || !pairs[0].Chosen.Completion[2] || !pairs[0].Rejected.Completion[2] {
		t.Fatalf("compiled pair=%+v", pairs)
	}
}
