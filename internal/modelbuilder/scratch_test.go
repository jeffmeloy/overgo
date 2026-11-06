package modelbuilder

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/scratchmodel"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

func TestScratchWorkflowSelection(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	if _, err := ScratchWorkflowSelection(t.Context(), store); err == nil {
		t.Fatal("absent profile admitted")
	}
	if _, err := scratchmodel.PublishDerivationProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	profile := testcheck.Must(scratchmodel.ResolveActiveDerivationProfile(t.Context(), store))(t)
	want := testcheck.Must(artifact.JSONID(artifact.KindRecipe, struct {
		Runtime string      `json:"runtime"`
		Profile artifact.ID `json:"profile"`
	}{"scratch-model-builder/v1", profile.ID}))(t)
	got, err := ScratchWorkflowSelection(t.Context(), store)
	if err != nil || got != want {
		t.Fatalf("selection identity changed: %s != %s, %v", got, want, err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(errors.New("selection cancelled"))
	if _, err := ScratchWorkflowSelection(ctx, store); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled selection = %v", err)
	}
}

func TestScratchBuilderPublishesCampaign(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	ctx := t.Context()
	if _, err := scratchmodel.PublishDerivationProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	session := testcheck.Must(NewScratchSession(ctx, ScratchRequest{
		Repository: store, Documents: []string{"abcd", "bcda", "cdab", "dabc"}, Seed: 1, Steps: 1,
	}))(t)
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "scratch-builder-operation")
	result := testcheck.Must(workflowruntime.ExecuteModelBuild(ctx, store, operation, session))(t)
	if result.DerivationProfile.Kind() != artifact.KindProfile {
		t.Fatal("model build omitted derivation profile authority")
	}
	for _, id := range []artifact.ID{result.Evaluation, result.Evidence, result.Decision} {
		if _, found, err := artifact.ReadContent(t.Context(), store, id); err != nil || !found {
			t.Fatalf("published artifact %s found=%t err=%v", id, found, err)
		}
	}
}
