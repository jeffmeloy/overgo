package modelbuilder

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/scratchmodel"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func TestScratchBuilderPublishesCampaign(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, err := scratchmodel.PublishDerivationProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	session, err := NewScratchSession(ctx, ScratchRequest{
		Repository: store, Documents: []string{"abcd", "bcda", "cdab", "dabc"}, Seed: 1, Steps: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "scratch-builder-operation")
	result, err := workflowruntime.ExecuteModelBuild(ctx, store, operation, session)
	if err != nil {
		t.Fatal(err)
	}
	if result.DerivationProfile.Kind() != artifact.KindProfile {
		t.Fatal("model build omitted derivation profile authority")
	}
	for _, id := range []artifact.ID{result.Evaluation, result.Evidence, result.Decision} {
		if _, found, err := artifact.ReadContent(t.Context(), store, id); err != nil || !found {
			t.Fatalf("published artifact %s found=%t err=%v", id, found, err)
		}
	}
}
