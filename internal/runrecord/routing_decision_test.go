package runrecord

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/invocation"
	"overgo/internal/testutil"
)

func TestRoutingDecisionContract(t *testing.T) {
	head := artifact.CommitID{1}
	first := routingCandidateFixture(t, "first", head, RouteMeasurement{
		Attempts: 4, Succeeded: 4, EvidenceUnits: 8, CostUnits: 2, CostObserved: true, LatencyNS: 40,
	})
	second := routingCandidateFixture(t, "second", head, RouteMeasurement{
		Attempts: 4, Succeeded: 3, EvidenceUnits: 4, CostUnits: 4, CostObserved: true, LatencyNS: 80,
	})
	base := RoutingDecision{
		Intent: "evidence-lookup", Effect: invocation.ClassInspection,
		Authority: testutil.ArtifactID(t, artifact.KindEvidence, "authority"), Head: head,
		Candidates: []RoutingCandidateObservation{second, first},
		// A caller cannot assert an outcome: the exact observations derive it.
		Disposition: RouteDecisionRequired, Winner: second.Selection,
	}

	decision, err := NewRoutingDecision(base)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Disposition != RouteSelected || decision.Winner != first.Selection {
		t.Fatalf("decision = (%s, %s), want derived winner %s", decision.Disposition, decision.Winner, first.Selection)
	}
	if artifact.CompareID(decision.Candidates[0].Selection, decision.Candidates[1].Selection) >= 0 {
		t.Fatalf("candidate order = [%s, %s], want canonical selection order", decision.Candidates[0].Selection, decision.Candidates[1].Selection)
	}
	reversed := base
	reversed.Candidates = []RoutingCandidateObservation{first, second}
	canonical, err := NewRoutingDecision(reversed)
	if err != nil {
		t.Fatal(err)
	}
	if canonical.ID != decision.ID || canonical.ValidateIdentity() != nil {
		t.Fatalf("canonical decision = %s, validation %v; want stable identity %s", canonical.ID, canonical.ValidateIdentity(), decision.ID)
	}
	parents := decision.Lineage()
	for _, candidate := range decision.Candidates {
		if slices.ContainsFunc(parents, func(edge artifact.Lineage) bool { return edge.Parent == candidate.Selection }) {
			t.Fatalf("derived selection %s became execution authority lineage", candidate.Selection)
		}
		for _, evidence := range []artifact.ID{candidate.Capability, candidate.Manual, candidate.Probe} {
			if !slices.ContainsFunc(parents, func(edge artifact.Lineage) bool { return edge.Parent == evidence }) {
				t.Fatalf("observed route fact %s is absent from lineage", evidence)
			}
		}
	}

	for _, test := range []struct {
		name   string
		mutate func(*RoutingDecision)
	}{
		{name: "candidate effect", mutate: func(value *RoutingDecision) { value.Candidates[0].Effect = invocation.ClassMutation }},
		{name: "candidate head", mutate: func(value *RoutingDecision) { value.Candidates[0].Head = artifact.CommitID{2} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			invalid := base
			invalid.Candidates = slices.Clone(base.Candidates)
			test.mutate(&invalid)
			if _, err := NewRoutingDecision(invalid); err == nil {
				t.Fatal("incoherent candidate was admitted")
			}
		})
	}
}

func routingCandidateFixture(t *testing.T, alias string, head artifact.CommitID, measurement RouteMeasurement) RoutingCandidateObservation {
	t.Helper()
	return RoutingCandidateObservation{
		Capability:  testutil.ArtifactID(t, artifact.KindProfile, alias+"-capability"),
		Manual:      testutil.ArtifactID(t, artifact.KindRecipe, alias+"-manual"),
		Selection:   testutil.ArtifactID(t, artifact.KindProfile, alias+"-selection"),
		Probe:       testutil.ArtifactID(t, artifact.KindEvidence, alias+"-probe"),
		Boundary:    invocation.BoundaryInternal,
		Effect:      invocation.ClassInspection,
		Head:        head,
		Rejection:   RouteEligible,
		Measurement: measurement,
	}
}
