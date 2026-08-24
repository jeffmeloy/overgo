package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestInteractionTraceIdentity(t *testing.T) {
	value := Interaction{
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "trace recipe"),
		Model:  testutil.ArtifactID(t, artifact.KindModel, "trace model"),
	}
	request := testutil.ArtifactID(t, artifact.KindEvidence, "trace request")
	messages := []InteractionMessage{
		{Role: "user", Content: "question"},
		{Role: "assistant", Content: "answer"},
	}
	first, err := NewInteractionTrace(value, request, messages, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewInteractionTrace(value, request, messages, nil)
	if err != nil || first.ID != second.ID || first.Events[0].Kind != InteractionEventRequest ||
		first.Events[1].Kind != InteractionEventOutput {
		t.Fatalf("trace identity = (%+v, %+v, %v)", first, second, err)
	}
}
