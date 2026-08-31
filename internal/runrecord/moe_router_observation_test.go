package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMoERouterObservationIdentityCoverageAndValidation(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	fixture := MoERouterObservation{
		Run: id(artifact.KindRun, "run"), Model: id(artifact.KindModel, "model"), Dataset: id(artifact.KindDataset, "dataset"),
		Split: id(artifact.KindDatasetShard, "split"), Recipe: id(artifact.KindRecipe, "recipe"), Code: id(artifact.KindEvidence, "code"),
		Checkpoint: id(artifact.KindCheckpoint, "checkpoint"), Policy: id(artifact.KindRecipe, "policy"), Step: 7, Layer: 3,
		Rows: 2, Experts: 3, TopK: 2, Selections: []uint32{0, 2, 1, 0},
		CombineWeights: []float32{0.7, 0.3, 0.6, 0.4}, Accepted: []bool{true, true, true, false},
		Margins: []MoERouterMargin{{Observed: true, Value: 0.2}, {Observed: true}},
	}
	observation, err := NewMoERouterObservation(fixture)
	if err != nil {
		t.Fatal(err)
	}
	content, err := observation.content()
	if err != nil || content.Descriptor.ID != observation.ID {
		t.Fatalf("observation content = (%s, %v)", content.Descriptor.ID, err)
	}
	repeated, err := NewMoERouterObservation(fixture)
	if err != nil || repeated.ID != observation.ID {
		t.Fatalf("repeated identity = (%s, %v), want %s", repeated.ID, err, observation.ID)
	}
	invalid := fixture
	invalid.Selections = invalid.Selections[:len(invalid.Selections)-1]
	if _, err := NewMoERouterObservation(invalid); err == nil {
		t.Fatal("incomplete selection coverage accepted")
	}
	invalid = fixture
	invalid.Margins = invalid.Margins[:1]
	if _, err := NewMoERouterObservation(invalid); err == nil {
		t.Fatal("missing margin coverage accepted")
	}
	invalid = fixture
	invalid.Selections[1] = invalid.Selections[0]
	if _, err := NewMoERouterObservation(invalid); err == nil {
		t.Fatal("duplicate row selection accepted")
	}
	invalid = fixture
	invalid.TopK = invalid.Experts
	invalid.Selections = []uint32{0, 1, 2, 2, 1, 0}
	invalid.CombineWeights = []float32{1, 1, 1, 1, 1, 1}
	invalid.Accepted = []bool{true, true, true, true, true, true}
	invalid.Margins = []MoERouterMargin{{}, {}}
	if _, err := NewMoERouterObservation(invalid); err != nil {
		t.Fatalf("all-expert route rejected: %v", err)
	}
}
