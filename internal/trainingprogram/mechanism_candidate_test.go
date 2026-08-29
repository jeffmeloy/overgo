package trainingprogram

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMarinCandidateAdmissionRequiresGapCostBenefitAndFalsifier(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	open := MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "quantile-router"), Owner: id(artifact.KindEvidence, "router-owner"),
		Gap: id(artifact.KindEvidence, "router-gap"), Falsifier: id(artifact.KindRecipe, "router-falsifier"), GapState: MechanismGapOpen,
	}
	closed := MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "existing-routing"), Owner: id(artifact.KindEvidence, "existing-owner"),
		Gap: id(artifact.KindEvidence, "no-gap"), Falsifier: id(artifact.KindRecipe, "owner-audit"), GapState: MechanismGapClosed,
	}
	census, err := CompileMechanismCensus([]MechanismAssessment{open, closed})
	if err != nil {
		t.Fatal(err)
	}
	provenance := id(artifact.KindEvidence, "provenance")
	cost := id(artifact.KindEvidence, "cost")
	benefit := id(artifact.KindEvidence, "benefit")
	candidate, err := CompileMechanismCandidate(census, provenance, open.Mechanism, cost, benefit)
	if err != nil {
		t.Fatal(err)
	}
	parents := make(map[artifact.ID]bool)
	for _, edge := range candidate.Lineage() {
		parents[edge.Parent] = true
	}
	for _, required := range []artifact.ID{census.ID(), provenance, open.Mechanism, open.Owner, open.Gap, cost, benefit, open.Falsifier} {
		if !parents[required] {
			t.Fatalf("candidate lineage omits %s", required)
		}
	}
	content, err := candidate.Content()
	if err != nil || content.Descriptor.ID != candidate.ID() {
		t.Fatalf("candidate content = (%s, %v)", content.Descriptor.ID, err)
	}
	if _, err := CompileMechanismCandidate(census, provenance, closed.Mechanism, cost, benefit); err == nil {
		t.Fatal("closed gap admitted")
	}
	if _, err := CompileMechanismCandidate(census, provenance, open.Mechanism, artifact.ID{}, benefit); err == nil {
		t.Fatal("missing cost evidence admitted")
	}
	if _, err := CompileMechanismCandidate(census, provenance, open.Mechanism, cost, cost); err == nil {
		t.Fatal("non-independent benefit evidence admitted")
	}
	if _, err := CompileMechanismCandidate(census, provenance, id(artifact.KindRecipe, "uncensused"), cost, benefit); err == nil {
		t.Fatal("uncensused mechanism admitted")
	}
}
