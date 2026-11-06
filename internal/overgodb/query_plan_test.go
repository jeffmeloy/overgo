package overgodb

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestProjectionQueryPlanEfficiency pins the deterministic query planner:
// kind, media, and schema lookups ride their owning projection index and
// inspect only that index's candidates, a media/schema pair rides the
// narrower index at this head, seeded lookups report the seed plan, an
// unowned contract may refuse instead of silently scanning the catalog, and
// every result reports inspected, matched, returned, and loaded facts.
func TestProjectionQueryPlanEfficiency(t *testing.T) {
	t.Parallel()
	store := testcheck.Must(Open(t.TempDir()))(t)
	defer store.Close()
	ctx := t.Context()
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

	// Each contract rides the index that owns it and inspects only what that index holds: the media index for its
	// media type and for a pair whose media is the narrower, the kind index for its kind, the schema index when the
	// head measures it narrower than the pair's media, and the seed for a named artifact.
	media := QueryPlanReport{Index: "media", Inspected: 2, Matched: 2, Returned: 2, Loaded: 2}
	for _, test := range []struct {
		name  string
		query Query
		want  QueryPlanReport
	}{
		{"media", Query{MediaType: "application/narrow"}, media},
		{"kind", Query{Kind: artifact.KindEvidence}, QueryPlanReport{Index: "kind", Inspected: 2, Matched: 2, Returned: 2, Loaded: 2}},
		{"narrower media", Query{MediaType: "application/narrow", Schema: "overgo/wide/v1"}, QueryPlanReport{Index: "media", Inspected: 2, Matched: 1, Returned: 1, Loaded: 1}},
		{"head-measured schema", Query{MediaType: "application/wide", Schema: "overgo/solo/v1"}, QueryPlanReport{Index: "schema", Inspected: 1}},
		{"seed", Query{Artifact: &narrowOne.ID}, QueryPlanReport{Index: "seed", Inspected: 1, Matched: 1, Returned: 1, Loaded: 1}},
	} {
		test.query.MaxResults, test.query.Projection = 10, ProjectArtifacts
		result := testcheck.Must(store.Query(ctx, test.query))(t)
		if result.Plan != test.want {
			t.Fatalf("%s plan = %+v, want %+v", test.name, result.Plan, test.want)
		}
	}

	unowned := Query{MaxResults: 100, Projection: ProjectArtifacts}
	scan := testcheck.Must(store.Query(ctx, unowned))(t)
	if scan.Plan.Index != "catalog" || scan.Plan.Inspected != 8 {
		t.Fatalf("catalog plan = %+v", scan.Plan)
	}
	unowned.RequireIndex = true
	if _, err := store.Query(ctx, unowned); err == nil ||
		!strings.Contains(err.Error(), "no projection index owns this query contract") {
		t.Fatalf("unowned contract was not refused: %v", err)
	}

	repeat := testcheck.Must(store.Query(ctx, Query{MediaType: "application/narrow", MaxResults: 10, Projection: ProjectArtifacts}))(t)
	if repeat.Plan != media {
		t.Fatalf("equal contract and head produced unequal plans: %+v vs %+v", repeat.Plan, media)
	}
}
