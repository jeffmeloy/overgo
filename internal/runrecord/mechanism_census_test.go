package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestMarinMechanismCensusBindsExistingOwnersAndFalsifiers(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	assessment := trainingprogram.MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "router-controller"), Owner: id(artifact.KindEvidence, "densecausal-owner"),
		Gap: id(artifact.KindEvidence, "missing-router-observations"), Falsifier: id(artifact.KindRecipe, "router-observation-audit"),
		GapState: trainingprogram.MechanismGapOpen,
	}
	census, err := trainingprogram.CompileMechanismCensus([]trainingprogram.MechanismAssessment{assessment})
	if err != nil {
		t.Fatal(err)
	}
	content, err := census.Content()
	if err != nil {
		t.Fatal(err)
	}
	descriptors := []artifact.Descriptor{{ID: assessment.Mechanism}, {ID: assessment.Owner}, {ID: assessment.Gap}, {ID: assessment.Falsifier}}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/mechanism-census", Artifacts: descriptors, Contents: []artifact.Content{content}, Lineage: census.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(t.Context(), census.ID())
	if err != nil || len(parents) != 4 {
		t.Fatalf("stored census parents = (%d, %v), want 4", len(parents), err)
	}
}
