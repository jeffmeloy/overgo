// Package composition_test verifies cross-package composition contracts.
package composition_test

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestCompositeCandidateBindsModelRoutingSteeringAndTransform(t *testing.T) {
	component := func(kind artifact.Kind, label string) artifact.ID {
		return testutil.ArtifactID(t, kind, label)
	}
	components := []modelrecipe.CandidateComponent{
		{Domain: modelrecipe.CandidateSteering, Specification: component(artifact.KindRecipe, "steering direction")},
		{Domain: modelrecipe.CandidateDatasetTransform, Specification: component(artifact.KindEvidence, "dataset transform")},
		{Domain: modelrecipe.CandidateModelPrototype, Specification: component(artifact.KindRecipe, "model prototype")},
		{Domain: modelrecipe.CandidateComposition, Specification: component(artifact.KindRecipe, "model composition")},
		{Domain: modelrecipe.CandidateRouting, Specification: component(artifact.KindRecipe, "routing target")},
	}
	subject := component(artifact.KindModelDefinition, "composite subject")
	parent := component(artifact.KindModel, "composite parent")
	spec := modelrecipe.CandidateSpec{
		Subject: subject, Parent: parent, Components: slices.Clone(components),
		Prediction: recipe.SteeringPrediction{
			Metric: "held-out composite quality", Benefit: 0.15, Cost: 20, Unit: "ratio", Uncertainty: 0.1,
		},
		Falsifier: component(artifact.KindRecipe, "composite falsifier"),
		References: []modelrecipe.CandidateReference{
			{Role: modelrecipe.CandidateReferenceBaseline, Subject: parent, Evidence: component(artifact.KindEvidence, "baseline")},
			{Role: modelrecipe.CandidateReferenceGap, Subject: subject, Evidence: component(artifact.KindEvidence, "gap")},
			{Role: modelrecipe.CandidateReferenceProvenance, Subject: subject, Evidence: component(artifact.KindEvidence, "prototype provenance")},
		},
		DevelopmentSplit:  component(artifact.KindDatasetShard, "development split"),
		PromotionSplit:    component(artifact.KindDatasetShard, "promotion split"),
		DevelopmentBudget: component(artifact.KindEvidence, "development budget"),
		PromotionBudget:   component(artifact.KindEvidence, "promotion budget"),
		Code:              component(artifact.KindEvidence, "code revision"),
		Environment:       component(artifact.KindEvidence, "environment"),
	}
	candidate, err := modelrecipe.NewCandidate(spec)
	if err != nil {
		t.Fatal(err)
	}

	bound := make(map[modelrecipe.CandidateDomain]artifact.ID)
	for _, declared := range candidate.Spec().Components {
		bound[declared.Domain] = declared.Specification
	}
	for _, declared := range components {
		if bound[declared.Domain] != declared.Specification {
			t.Fatalf("composite omitted or replaced %s component", declared.Domain)
		}
	}
	parents := make(map[artifact.ID]bool)
	for _, edge := range candidate.Lineage() {
		parents[edge.Parent] = true
	}
	for _, declared := range components {
		if !parents[declared.Specification] {
			t.Fatalf("component %s absent from candidate lineage", declared.Domain)
		}
	}

	shuffled := spec
	shuffled.Components = slices.Clone(spec.Components)
	shuffled.References = slices.Clone(spec.References)
	slices.Reverse(shuffled.Components)
	slices.Reverse(shuffled.References)
	canonical, err := modelrecipe.NewCandidate(shuffled)
	if err != nil {
		t.Fatal(err)
	}
	if canonical.ID() != candidate.ID() {
		t.Fatal("composite identity depends on declaration order")
	}

	retag := map[modelrecipe.CandidateDomain]modelrecipe.CandidateDomain{
		modelrecipe.CandidateModelPrototype:   modelrecipe.CandidateMechanism,
		modelrecipe.CandidateComposition:      modelrecipe.CandidateRouting,
		modelrecipe.CandidateRouting:          modelrecipe.CandidateSteering,
		modelrecipe.CandidateSteering:         modelrecipe.CandidateModelPrototype,
		modelrecipe.CandidateDatasetTransform: modelrecipe.CandidateCode,
	}
	for index, declared := range components {
		t.Run(string(declared.Domain), func(t *testing.T) {
			dropped := cloneCompositeSpec(spec)
			dropped.Components = append(dropped.Components[:index], dropped.Components[index+1:]...)
			without, err := modelrecipe.NewCandidate(dropped)
			if err != nil {
				t.Fatal(err)
			}
			if without.ID() == candidate.ID() {
				t.Fatal("dropping a component preserved candidate identity")
			}

			retyped := cloneCompositeSpec(spec)
			retyped.Components[index].Domain = retag[declared.Domain]
			changedType, err := modelrecipe.NewCandidate(retyped)
			if err != nil {
				t.Fatal(err)
			}
			if changedType.ID() == candidate.ID() {
				t.Fatal("retyping a component preserved candidate identity")
			}

			replaced := cloneCompositeSpec(spec)
			replaced.Components[index].Specification = component(declared.Specification.Kind(), "replacement "+string(declared.Domain))
			changedSpec, err := modelrecipe.NewCandidate(replaced)
			if err != nil {
				t.Fatal(err)
			}
			if changedSpec.ID() == candidate.ID() {
				t.Fatal("replacing a component preserved candidate identity")
			}
		})
	}
}

func cloneCompositeSpec(value modelrecipe.CandidateSpec) modelrecipe.CandidateSpec {
	value.Components = slices.Clone(value.Components)
	value.References = slices.Clone(value.References)
	return value
}
