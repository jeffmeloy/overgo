package runrecord

import (
	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"testing"
)

func TestAgentMutationCheckpointIdentity(t *testing.T) {
	preimage := testutil.ArtifactID(t, artifact.KindFile, "before")
	postimage := testutil.ArtifactID(t, artifact.KindFile, "after")
	checkpoint, err := NewAgentMutationCheckpoint(AgentMutationCheckpoint{
		Operation: testutil.ArtifactID(t, artifact.KindEvidence, "operation"), MutationEpoch: 2,
		Entries: []AgentCheckpointEntry{
			{Target: "internal/a.go", Mode: "0644", Encoding: "utf-8", Preimage: &preimage, ExpectedPostimage: &postimage},
			{Target: "internal/new.go", Mode: "absent", Encoding: "binary", Gap: "new target has no preimage"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if checkpoint.ID.Kind() != artifact.KindCheckpoint || checkpoint.Captured != 1 || checkpoint.Gaps != 1 || len(checkpoint.Lineage()) == 0 {
		t.Fatalf("checkpoint = %+v", checkpoint)
	}
	content, err := checkpoint.Content()
	if err != nil {
		t.Fatal(err)
	}
	if len(content.Data) == 0 {
		t.Fatal("checkpoint content absent")
	}
}
