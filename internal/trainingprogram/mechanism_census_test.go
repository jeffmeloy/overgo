package trainingprogram

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMarinMechanismCensusBindsExistingOwnersAndFalsifiers(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	router := MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "quantile-router"), Owner: id(artifact.KindEvidence, "densecausal-router-owner"),
		Gap: id(artifact.KindEvidence, "router-observation-gap"), Falsifier: id(artifact.KindRecipe, "paired-router-evaluation"),
		GapState: MechanismGapOpen,
	}
	scaling := MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "isoflop-study"), Owner: id(artifact.KindRecipe, "evaluation-owner"),
		Gap: id(artifact.KindEvidence, "scaling-accounting-covered"), Falsifier: id(artifact.KindRecipe, "scaling-owner-audit"),
		GapState: MechanismGapClosed,
	}
	census, err := CompileMechanismCensus([]MechanismAssessment{router, scaling})
	if err != nil {
		t.Fatal(err)
	}
	reordered, err := CompileMechanismCensus([]MechanismAssessment{scaling, router})
	if err != nil || reordered.ID() != census.ID() {
		t.Fatalf("reordered census = (%s, %v), want %s", reordered.ID(), err, census.ID())
	}
	assessments := census.Assessments()
	if len(assessments) != 2 || assessments[0].Mechanism.String() > assessments[1].Mechanism.String() {
		t.Fatalf("assessments are not canonical: %+v", assessments)
	}
	assessments[0].GapState = "mutated"
	if census.Assessments()[0].GapState == "mutated" {
		t.Fatal("assessment view aliases census state")
	}
	parents := make(map[artifact.ID]bool)
	for _, edge := range census.Lineage() {
		parents[edge.Parent] = true
	}
	for _, assessment := range []MechanismAssessment{router, scaling} {
		for _, required := range []artifact.ID{assessment.Mechanism, assessment.Owner, assessment.Gap, assessment.Falsifier} {
			if !parents[required] {
				t.Fatalf("lineage omits %s", required)
			}
		}
	}
	content, err := census.Content()
	if err != nil || content.Descriptor.ID != census.ID() || content.Descriptor.Schema != MechanismCensusSchema {
		t.Fatalf("content = (%s, %q, %v)", content.Descriptor.ID, content.Descriptor.Schema, err)
	}

	invalid := router
	invalid.Owner = artifact.ID{}
	if _, err := CompileMechanismCensus([]MechanismAssessment{invalid}); err == nil {
		t.Fatal("missing owner accepted")
	}
	invalid = router
	invalid.Falsifier = id(artifact.KindEvidence, "not-a-falsifier-recipe")
	if _, err := CompileMechanismCensus([]MechanismAssessment{invalid}); err == nil {
		t.Fatal("non-recipe falsifier accepted")
	}
	invalid = router
	invalid.GapState = "assumed"
	if _, err := CompileMechanismCensus([]MechanismAssessment{invalid}); err == nil {
		t.Fatal("unknown gap state accepted")
	}
	if _, err := CompileMechanismCensus([]MechanismAssessment{router, router}); err == nil {
		t.Fatal("duplicate mechanism accepted")
	}
	if _, err := CompileMechanismCensus(nil); err == nil {
		t.Fatal("empty census accepted")
	}
}
