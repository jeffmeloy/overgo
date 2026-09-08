package speechactivity

import (
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestRequireActivitySegmentsForComposition(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	result := recipecontract.ActivitySegments{Source: recipecontract.AudioReference{Audio: testutil.ArtifactID(t, artifact.KindFile, "audio"), Profile: testutil.ArtifactID(t, artifact.KindProfile, "format")}, Segments: []recipecontract.ActivitySegment{{Span: recipecontract.SampleSpan{Start: 10, End: 20}, Confidence: .75}}}
	content, err := artifact.JSONContent(activityContract, result)
	if err != nil {
		t.Fatal(err)
	}
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
	wrong, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "overgo/wrong-output/v1"), result)
	if err != nil {
		t.Fatal(err)
	}
	// Different bytes keep the deliberately wrong schema from colliding with
	// the already-bound content identity.
	wrong.Data = append(wrong.Data, ' ')
	wrong.Descriptor.ID, err = artifact.IdentifyBytes(artifact.KindOutput, wrong.Data)
	if err != nil {
		t.Fatal(err)
	}
	wrong.Descriptor.Size = uint64(len(wrong.Data))
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "wrong", Contents: []artifact.Content{wrong}}); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireSegments(t.Context(), store, wrong.Descriptor.ID); err == nil {
		t.Fatal("foreign schema accepted")
	}
}
