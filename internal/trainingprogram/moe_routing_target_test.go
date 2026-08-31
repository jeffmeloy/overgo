package trainingprogram

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMoERoutingTargetRequiresDeclaredPriorStrataAndReopenTriggers(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	candidate := routingTargetCandidate(t, id)
	prior := id(artifact.KindEvidence, "routing-prior")
	first := id(artifact.KindDatasetShard, "language")
	second := id(artifact.KindDatasetShard, "code")
	assumption := moeRoutingAssumption{
		Name: "stable-stratum-demand", Statement: "The declared target remains useful within each named stratum.",
		Evidence: id(artifact.KindEvidence, "target-evidence"), ReopenTrigger: id(artifact.KindRecipe, "target-drift-check"),
	}

	uniform, err := moeRoutingTargetContract.New(moeRoutingTarget{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(),
		moeRoutingTargetSpec: moeRoutingTargetSpec{
			Mode: moeRoutingTargetUniform, Prior: prior,
			Strata:      []moeRoutingStratumTarget{{Stratum: first, Masses: []uint64{1, 1}}, {Stratum: second, Masses: []uint64{3, 3}}},
			Assumptions: []moeRoutingAssumption{assumption},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := moeRoutingTargetContract.Content(uniform)
	if err != nil || content.Descriptor.ID != uniform.ID || content.Descriptor.Schema != moeRoutingTargetSchema {
		t.Fatalf("uniform content = (%+v, %v)", content.Descriptor, err)
	}
	parents := make(map[artifact.ID]bool)
	for _, edge := range moeRoutingTargetLineage(uniform) {
		parents[edge.Parent] = true
	}
	for _, required := range []artifact.ID{candidate.ID(), prior, first, second, assumption.Evidence, assumption.ReopenTrigger} {
		if !parents[required] {
			t.Fatalf("routing target lineage omits %s", required)
		}
	}

	for _, mode := range []moeRoutingTargetMode{moeRoutingTargetDeclaredPrior, moeRoutingTargetCapacityWeighted} {
		if _, err := moeRoutingTargetContract.New(moeRoutingTarget{
			Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(),
			moeRoutingTargetSpec: moeRoutingTargetSpec{
				Mode: mode, Prior: prior, Strata: []moeRoutingStratumTarget{{Stratum: first, Masses: []uint64{1, 4}}},
				Assumptions: []moeRoutingAssumption{assumption},
			},
		}); err != nil {
			t.Fatalf("explicit %s target rejected: %v", mode, err)
		}
	}

	valid := moeRoutingTargetSpec{
		Mode: moeRoutingTargetDeclaredPrior, Prior: prior,
		Strata:      []moeRoutingStratumTarget{{Stratum: first, Masses: []uint64{1, 4}}},
		Assumptions: []moeRoutingAssumption{assumption},
	}
	cases := map[string]func(*moeRoutingTargetSpec){
		"implicit mode":       func(spec *moeRoutingTargetSpec) { spec.Mode = "" },
		"missing prior":       func(spec *moeRoutingTargetSpec) { spec.Prior = artifact.ID{} },
		"missing strata":      func(spec *moeRoutingTargetSpec) { spec.Strata = nil },
		"missing assumptions": func(spec *moeRoutingTargetSpec) { spec.Assumptions = nil },
		"missing reopen":      func(spec *moeRoutingTargetSpec) { spec.Assumptions[0].ReopenTrigger = artifact.ID{} },
		"inconsistent experts": func(spec *moeRoutingTargetSpec) {
			spec.Strata = append(spec.Strata, moeRoutingStratumTarget{Stratum: second, Masses: []uint64{1}})
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			spec := cloneMoERoutingTargetSpec(valid)
			mutate(&spec)
			if _, err := moeRoutingTargetContract.New(moeRoutingTarget{
				Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), moeRoutingTargetSpec: spec,
			}); err == nil {
				t.Fatal("invalid routing target admitted")
			}
		})
	}
	invalidUniform := cloneMoERoutingTargetSpec(valid)
	invalidUniform.Mode = moeRoutingTargetUniform
	if _, err := moeRoutingTargetContract.New(moeRoutingTarget{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), moeRoutingTargetSpec: invalidUniform,
	}); err == nil {
		t.Fatal("unequal masses admitted as a uniform target")
	}
}

func routingTargetCandidate(t *testing.T, id func(artifact.Kind, string) artifact.ID) MechanismCandidate {
	t.Helper()
	assessment := MechanismAssessment{
		Mechanism: id(artifact.KindRecipe, "quantile-router"), Owner: id(artifact.KindEvidence, "router-owner"),
		Gap: id(artifact.KindEvidence, "router-gap"), Falsifier: id(artifact.KindRecipe, "router-falsifier"), GapState: MechanismGapOpen,
	}
	census, err := CompileMechanismCensus([]MechanismAssessment{assessment})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := CompileMechanismCandidate(
		census, id(artifact.KindEvidence, "marin-provenance"), assessment.Mechanism,
		id(artifact.KindEvidence, "cost-bound"), id(artifact.KindEvidence, "benefit-evidence"),
	)
	if err != nil {
		t.Fatal(err)
	}
	return candidate
}
