package modelbuilder

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func TestScratchBuilderPublishesCampaign(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	session, err := NewScratchSession(ScratchRequest{
		Repository: store, Documents: []string{"abcd", "bcda", "cdab", "dabc"}, Seed: 1, Steps: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "scratch-builder-operation")
	result, err := workflowruntime.ExecuteModelBuild(context.Background(), store, operation, session)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []artifact.ID{result.Evaluation, result.Evidence, result.Decision} {
		if _, found, err := store.Content(context.Background(), id); err != nil || !found {
			t.Fatalf("published artifact %s found=%t err=%v", id, found, err)
		}
	}
}
