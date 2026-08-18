package trainingdata

import (
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
