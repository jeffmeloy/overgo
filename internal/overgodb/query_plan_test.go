package overgodb

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestProjectionQueryPlanEfficiency pins the deterministic query planner:
// kind, media, and schema lookups ride their owning projection index and
// inspect only that index's candidates, a media/schema pair rides the
// narrower index at this head, seeded lookups report the seed plan, an
// unowned contract may refuse instead of silently scanning the catalog, and
// every result reports inspected, matched, returned, and loaded facts.
func TestProjectionQueryPlanEfficiency(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	wide := make([]artifact.Descriptor, 0, 8)
	for _, name := range []string{"a", "b", "c", "d", "e", "f"} {
		wide = append(wide, artifact.Descriptor{
			ID: testutil.ArtifactID(t, artifact.KindDataset, "dataset-"+name), Size: 1,
			MediaType: "application/wide", Schema: "overgo/wide/v1",
		})
	}
	narrowOne := artifact.Descriptor{
		ID: testutil.ArtifactID(t, artifact.KindEvidence, "narrow-one"), Size: 1,
		MediaType: "application/narrow", Schema: "overgo/wide/v1",
	}
	narrowTwo := artifact.Descriptor{
		ID: testutil.ArtifactID(t, artifact.KindEvidence, "narrow-two"), Size: 1,
		MediaType: "application/narrow", Schema: "overgo/solo/v1",
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/query-plan/base", Artifacts: append(wide, narrowOne, narrowTwo),
	}); err != nil {
		t.Fatal(err)
	}

	media, err := store.Query(ctx, Query{MediaType: "application/narrow", MaxResults: 10, Projection: ProjectArtifacts})
	if err != nil {
		t.Fatal(err)
	}
	if media.Plan.Index != "media" || media.Plan.Inspected != 2 || media.Plan.Matched != 2 ||
		media.Plan.Returned != 2 || media.Plan.Loaded != 2 {
		t.Fatalf("media plan = %+v", media.Plan)
	}
	kind, err := store.Query(ctx, Query{Kind: artifact.KindEvidence, MaxResults: 10, Projection: ProjectArtifacts})
	if err != nil {
		t.Fatal(err)
	}
	if kind.Plan.Index != "kind" || kind.Plan.Inspected != 2 || kind.Plan.Returned != 2 {
		t.Fatalf("kind plan scans beyond its projection: %+v", kind.Plan)
	}
	pair, err := store.Query(ctx, Query{
		MediaType: "application/narrow", Schema: "overgo/wide/v1", MaxResults: 10, Projection: ProjectArtifacts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if pair.Plan.Index != "media" || pair.Plan.Inspected != 2 || pair.Plan.Matched != 1 || pair.Plan.Returned != 1 {
		t.Fatalf("pair plan did not ride the narrower index: %+v", pair.Plan)
	}
	solo, err := store.Query(ctx, Query{
		MediaType: "application/wide", Schema: "overgo/solo/v1", MaxResults: 10, Projection: ProjectArtifacts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if solo.Plan.Index != "schema" || solo.Plan.Inspected != 1 || solo.Plan.Matched != 0 {
		t.Fatalf("pair plan tie-break is not head-measured: %+v", solo.Plan)
	}
	seeded, err := store.Query(ctx, Query{
		Artifact: &narrowOne.ID, MaxResults: 10, Projection: ProjectArtifacts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if seeded.Plan.Index != "seed" || seeded.Plan.Returned != 1 {
		t.Fatalf("seed plan = %+v", seeded.Plan)
	}

	unowned := Query{MaxResults: 100, Projection: ProjectArtifacts}
	scan, err := store.Query(ctx, unowned)
	if err != nil {
		t.Fatal(err)
	}
	if scan.Plan.Index != "catalog" || scan.Plan.Inspected != 8 {
		t.Fatalf("catalog plan = %+v", scan.Plan)
	}
	unowned.RequireIndex = true
	if _, err := store.Query(ctx, unowned); err == nil ||
		!strings.Contains(err.Error(), "no projection index owns this query contract") {
		t.Fatalf("unowned contract was not refused: %v", err)
	}

	repeat, err := store.Query(ctx, Query{MediaType: "application/narrow", MaxResults: 10, Projection: ProjectArtifacts})
	if err != nil {
		t.Fatal(err)
	}
	if repeat.Plan != media.Plan {
		t.Fatalf("equal contract and head produced unequal plans: %+v vs %+v", repeat.Plan, media.Plan)
	}
}
