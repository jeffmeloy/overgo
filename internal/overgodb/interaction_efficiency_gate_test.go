package overgodb_test

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestKindIndexInspectsLessThanScan proves a storage optimization claim
// from measured query plans: the same catalog read served by its owning
// kind index inspects a fraction of what the catalog scan inspects.
func TestKindIndexInspectsLessThanScan(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
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
}
