package capabilityruntime

import (
	"context"
	"runtime"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestExactCapabilityPlacementAndReceipt(t *testing.T) {
	store, modelID, program := capabilityFixture(t, "exact-capability-placement")
	selection := modelrecipetest.CandidateExecution(t, store, program)
	schema := testutil.ArtifactID(t, artifact.KindProfile, "exact-model-schema")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "capability/placement/schema", Artifacts: []artifact.Descriptor{{ID: schema}},
	}); err != nil {
		t.Fatal(err)
	}
	capability, err := (runrecord.CapabilityIdentity{
		Implementation: modelID, Release: "1.0.0",
		Transport: runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportBuiltin, Protocol: "overgo-model/1"},
		Schema:    schema, Platform: runrecord.CapabilityPlatform{OS: runtime.GOOS, Arch: runtime.GOARCH},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 4096, MaxOutputBytes: 4096, MaxConcurrent: 1,
			CPUThreads: 1, HostBytes: selection.Resources.ArtifactBytes,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	content, err := capability.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("capability/placement", []artifact.Content{content}, capability.Lineage(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	placement, err := ResolveExactCapabilityPlacement(t.Context(), store, capability.ID, selection)
	if err != nil || placement.Capability.Implementation != modelID {
		t.Fatalf("placement = (%+v, %v)", placement, err)
	}
	catalog := ExecutorCatalog{scalarModule: func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
		return raw, nil
	}}
	if _, err := catalog.ExecutePlaced(t.Context(), store, "model", placement, `{}`); err != nil {
		t.Fatal(err)
	}
	drifted := placement
	drifted.Selection.Identity = testutil.ArtifactID(t, artifact.KindProfile, "drifted-selection")
	if _, err := catalog.ExecutePlaced(t.Context(), store, "model", drifted, `{}`); err == nil {
		t.Fatal("execution admitted a model selection differing from its exact capability placement")
	}
}
