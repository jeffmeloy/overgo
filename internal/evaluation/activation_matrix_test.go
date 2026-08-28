package evaluation

import (
	"context"
	"runtime"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestActivationMatrixCoverage(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	implementation := testutil.ArtifactID(t, artifact.KindFile, "matrix-implementation")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "matrix-schema")
	contract := testutil.ArtifactID(t, artifact.KindRecipe, "matrix-contract")
	input := testutil.ArtifactID(t, artifact.KindDataset, "matrix-input")
	check := testutil.ArtifactID(t, artifact.KindProfile, "matrix-check")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "matrix-evidence")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "matrix/parents", Artifacts: []artifact.Descriptor{
		{ID: implementation}, {ID: schema}, {ID: contract}, {ID: input}, {ID: check}, {ID: evidence},
	}}); err != nil {
		t.Fatal(err)
	}
	capability, err := (runrecord.CapabilityIdentity{
		Implementation: implementation, Release: "1.0.0",
		Transport: runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportBuiltin, Protocol: "matrix/1"},
		Schema:    schema, Platform: runrecord.CapabilityPlatform{OS: runtime.GOOS, Arch: runtime.GOARCH},
		Resources: runrecord.CapabilityResourceEnvelope{MaxInputBytes: 1, MaxOutputBytes: 1, MaxConcurrent: 1, CPUThreads: 1, HostBytes: 1},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	content, err := capability.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("matrix/capability", []artifact.Content{content}, capability.Lineage(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	registry, _, err := PublishActivationCaseRegistry(ctx, store, []ActivationCase{{
		Name: "generate", Task: recipe.TaskGeneration, Contract: contract, Input: input, EvidenceCheck: check,
	}})
	if err != nil {
		t.Fatal(err)
	}
	local, err := NewActivationProfile(ActivationProfile{Name: "local", Kind: ActivationProfileLocalModel, Capability: capability.ID, Tasks: []recipe.Task{recipe.TaskGeneration}})
	if err != nil {
		t.Fatal(err)
	}
	peer, err := NewActivationProfile(ActivationProfile{Name: "peer", Kind: ActivationProfilePeerModel, Capability: capability.ID, Tasks: []recipe.Task{recipe.TaskGeneration}})
	if err != nil {
		t.Fatal(err)
	}
	matrix, _, err := PublishActivationMatrix(ctx, store, []ActivationProfile{peer, local}, []ActivationCoverageResult{{
		Case: registry.Cases[0], Profile: local.ID, Evidence: evidence,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if matrix.Denominator != 2 || matrix.Covered != 1 {
		t.Fatalf("matrix totals = %+v", matrix)
	}
	if err := AdmitActivationProfile(matrix, local.ID); err != nil {
		t.Fatal(err)
	}
	if err := AdmitActivationProfile(matrix, peer.ID); err == nil {
		t.Fatal("profile with a coverage gap was admitted")
	}
	foreign := testutil.ArtifactID(t, artifact.KindProfile, "unenumerated-profile")
	if err := AdmitActivationProfile(matrix, foreign); err == nil {
		t.Fatal("unenumerated activation profile was admitted")
	}
}
