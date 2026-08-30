package overgodb_test

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestInteractionEfficiencyGate proves a storage optimization claim from
// measured query plans: the same catalog read served by its owning kind
// index inspects a fraction of what the catalog scan inspects, the plan
// reports are the exact counters, and the claim wins the gate on the closed
// counter surface without shifting work anywhere else.
func TestInteractionEfficiencyGate(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	descriptors := make([]artifact.Descriptor, 0, 9)
	for _, name := range []string{"a", "b", "c", "d", "e", "f", "g"} {
		descriptors = append(descriptors, artifact.Descriptor{
			ID: testutil.ArtifactID(t, artifact.KindDataset, "gate-dataset-"+name), Size: 1,
		})
	}
	target := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "gate-target"), Size: 1}
	descriptors = append(descriptors, target)
	if _, err := store.Commit(ctx, artifact.Batch{Key: "gate/catalog", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}

	scan, err := store.Query(ctx, overgodb.Query{MaxResults: 100, Projection: overgodb.ProjectArtifacts})
	if err != nil || scan.Plan.Index != "catalog" {
		t.Fatalf("baseline scan = (%+v, %v)", scan.Plan, err)
	}
	indexed, err := store.Query(ctx, overgodb.Query{Kind: artifact.KindEvidence, MaxResults: 100, Projection: overgodb.ProjectArtifacts})
	if err != nil || indexed.Plan.Index != "kind" {
		t.Fatalf("candidate lookup = (%+v, %v)", indexed.Plan, err)
	}
	if indexed.Plan.Inspected >= scan.Plan.Inspected {
		t.Fatalf("index inspected %d, scan inspected %d", indexed.Plan.Inspected, scan.Plan.Inspected)
	}

	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "gate-trace-evidence")
	baseline := runrecord.EfficiencyTrace{
		Surface: runrecord.SurfaceStorage, Task: "resolve-evidence-artifact",
		Work: runrecord.InteractionWork{
			ScannedFacts: uint64(scan.Plan.Inspected), ReturnedFacts: uint64(scan.Plan.Loaded), ArtifactReads: 1,
		},
		Result: target.ID, Evidence: evidence,
	}
	candidate := baseline
	candidate.Work.ScannedFacts = uint64(indexed.Plan.Inspected)
	candidate.Work.ReturnedFacts = uint64(indexed.Plan.Loaded)
	comparison, err := runrecord.CompareEfficiencyTraces(
		candidate, baseline, runrecord.EfficiencyCounterNames(), nil,
	)
	if err != nil || !comparison.Win || len(comparison.Worsened) != 0 {
		t.Fatalf("measured storage claim = (%+v, %v)", comparison, err)
	}
}
