package main

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestRetainedProjectionPublications(t *testing.T) {
	var projections []struct {
		Name     string      `json:"name"`
		Evidence artifact.ID `json:"evidence"`
		Suite    artifact.ID `json:"suite"`
	}
	root := testutil.RepoRoot(t)
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/verification/projection_publications.json"), &projections); err != nil {
		t.Fatal(err)
	}
	if len(projections) == 0 {
		t.Fatal("projection publications are empty")
	}
	fixtures := retainedPublicationFixtures(t)
	for _, projection := range projections {
		t.Run(projection.Name, func(t *testing.T) {
			index := slices.IndexFunc(fixtures, func(f retainedPublicationFixture) bool { return f.Name == projection.Name })
			if index < 0 {
				t.Fatal("model publication absent")
			}
			checkRetainedModelPublication(t, fixtures[index])
			var spec verificationSpecification
			if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/verification", fixtures[index].Specification), &spec); err != nil {
				t.Fatal(err)
			}
			if !slices.ContainsFunc(spec.Claims, func(c runrecord.CapabilityClaim) bool {
				return c.Capability == "projection-grounded-colors-positions" && c.Tier == runrecord.TierExactGolden && c.WallNS > 0 && slices.Contains(c.Evidence, projection.Evidence)
			}) {
				t.Fatal("grounded projection claim absent")
			}
		})
	}
}
