package speechactivity

import (
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestRequireActivitySegmentsForComposition(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	result := recipecontract.ActivitySegments{Source: recipecontract.AudioReference{Audio: testutil.ArtifactID(t, artifact.KindFile, "audio"), Profile: testutil.ArtifactID(t, artifact.KindProfile, "format")}, Segments: []recipecontract.ActivitySegment{{Span: recipecontract.SampleSpan{Start: 10, End: 20}, Confidence: .75}}}
	content := testcheck.Must(artifact.JSONContent(activityContract, result))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "activity", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	actual, err := RequireSegments(t.Context(), store, content.Descriptor.ID)
	if err != nil || !reflect.DeepEqual(actual, result) {
		t.Fatalf("composition read: %+v %v", actual, err)
	}
	if _, err := RequireSegments(t.Context(), store, testutil.ArtifactID(t, artifact.KindOutput, "missing")); err == nil {
		t.Fatal("missing segments accepted")
	}
	wrong := testcheck.Must(artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "overgo/wrong-output/v1"), result))(t)
	// Different bytes keep the deliberately wrong schema from colliding with
	// the already-bound content identity.
	wrong.Data = append(wrong.Data, ' ')
	wrong.Descriptor.ID = testcheck.Must(artifact.IdentifyBytes(artifact.KindOutput, wrong.Data))(t)
	wrong.Descriptor.Size = uint64(len(wrong.Data))
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "wrong", Contents: []artifact.Content{wrong}}); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireSegments(t.Context(), store, wrong.Descriptor.ID); err == nil {
		t.Fatal("foreign schema accepted")
	}
}
