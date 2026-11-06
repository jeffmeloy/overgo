package evaluation

import (
	"runtime"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestActivationMatrixCoverage(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
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
	capability := testcheck.Must((runrecord.CapabilityIdentity{
		Implementation: implementation, Release: "1.0.0",
		Transport: runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportBuiltin, Protocol: "matrix/1"},
		Schema:    schema, Platform: runrecord.CapabilityPlatform{OS: runtime.GOOS, Arch: runtime.GOARCH},
		Resources: runrecord.CapabilityResourceEnvelope{MaxInputBytes: 1, MaxOutputBytes: 1, MaxConcurrent: 1, CPUThreads: 1, HostBytes: 1},
	}).Identify())(t)
	content := testcheck.Must(capability.Content())(t)
	batch := testcheck.Must(artifact.NewDocumentBatch("matrix/capability", []artifact.Content{content}, capability.Lineage(), nil))(t)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	registry, _, err := PublishActivationCaseRegistry(ctx, store, []ActivationCase{{
		Name: "generate", Task: recipe.TaskGeneration, Contract: contract, Input: input, EvidenceCheck: check,
	}})
	testcheck.NoError(t, err)
	local := testcheck.Must(NewActivationProfile(ActivationProfile{Name: "local", Kind: ActivationProfileLocalModel, Capability: capability.ID, Tasks: []recipe.Task{recipe.TaskGeneration}}))(t)
	peer := testcheck.Must(NewActivationProfile(ActivationProfile{Name: "peer", Kind: activationProfilePeerModel, Capability: capability.ID, Tasks: []recipe.Task{recipe.TaskGeneration}}))(t)
	matrix, _, err := PublishActivationMatrix(ctx, store, []ActivationProfile{peer, local}, []ActivationCoverageResult{{
		Case: registry.Cases[0], Profile: local.ID, Evidence: evidence,
	}})
	testcheck.NoError(t, err)
	if matrix.Denominator != 2 || matrix.Covered != 1 {
		t.Fatalf("matrix totals = %+v", matrix)
	}
	testcheck.NoError(t, AdmitActivationProfile(matrix, local.ID))
	if err := AdmitActivationProfile(matrix, peer.ID); err == nil {
		t.Fatal("profile with a coverage gap was admitted")
	}
	foreign := testutil.ArtifactID(t, artifact.KindProfile, "unenumerated-profile")
	if err := AdmitActivationProfile(matrix, foreign); err == nil {
		t.Fatal("unenumerated activation profile was admitted")
	}
}
