package modelrecipe

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
)

// AdmissionFacts projects the opaque candidate into the admission owner's
// package-neutral immutable contract.
func (value Candidate) AdmissionFacts() runrecord.CandidateAdmissionFacts {
	spec := value.Spec()
	components := make([]runrecord.CandidateAdmissionComponent, len(spec.Components))
	for index, component := range spec.Components {
		components[index] = runrecord.CandidateAdmissionComponent{
			Domain: string(component.Domain), Specification: component.Specification,
		}
	}
	references := make([]runrecord.CandidateAdmissionReference, len(spec.References))
	for index, reference := range spec.References {
		references[index] = runrecord.CandidateAdmissionReference{
			Role: string(reference.Role), Subject: reference.Subject, Evidence: reference.Evidence,
		}
	}
	return runrecord.CandidateAdmissionFacts{
		Candidate: value.ID(), Subject: spec.Subject, Parent: spec.Parent, Components: components,
		PredictionMetric: spec.Prediction.Metric, PredictedBenefit: spec.Prediction.Benefit,
		PredictedCost: spec.Prediction.Cost, BenefitUnit: spec.Prediction.Unit, CostUnit: spec.CostUnit,
		Falsifier: spec.Falsifier, References: references,
		DevelopmentSplit: spec.DevelopmentSplit, PromotionSplit: spec.PromotionSplit,
		DevelopmentBudget: spec.DevelopmentBudget, PromotionBudget: spec.PromotionBudget,
		Code: spec.Code, Environment: spec.Environment,
	}
}

// CandidateAdmissionAdapters returns the closed core Go adapter set. Later
// domains extend the caller's explicit set; no filesystem or runtime registry exists.
func CandidateAdmissionAdapters() []runrecord.CandidateComponentAdmissionAdapter {
	return []runrecord.CandidateComponentAdmissionAdapter{
		modelPrototypeAdmissionAdapter{}, datasetTransformAdmissionAdapter{}, codeAdmissionAdapter{},
	}
}

type modelPrototypeAdmissionAdapter struct{}

// CandidateDomain names the exact component domain this adapter closes.
func (modelPrototypeAdmissionAdapter) CandidateDomain() string {
	return string(CandidateModelPrototype)
}

// ValidateCandidateComponent requires the stored compiled-profile prototype.
func (modelPrototypeAdmissionAdapter) ValidateCandidateComponent(
	ctx context.Context,
	reader artifact.Reader,
	_ runrecord.CandidateAdmissionFacts,
	component runrecord.CandidateAdmissionComponent,
) error {
	_, err := RequireModelPrototype(ctx, reader, component.Specification)
	return err
}

type codeAdmissionAdapter struct{}

type datasetTransformAdmissionAdapter struct{}

// CandidateDomain names the dataset owner's immutable transform contract.
func (datasetTransformAdmissionAdapter) CandidateDomain() string {
	return string(CandidateDatasetTransform)
}

// ValidateCandidateComponent binds the transform owner's exact stored facts
// to the common candidate subject and execution evidence.
func (datasetTransformAdmissionAdapter) ValidateCandidateComponent(
	ctx context.Context,
	reader artifact.Reader,
	facts runrecord.CandidateAdmissionFacts,
	component runrecord.CandidateAdmissionComponent,
) error {
	transform, err := dataset.RequireDatasetTransform(ctx, reader, component.Specification)
	if err != nil {
		return err
	}
	spec := transform.Spec()
	if spec.Code != facts.Code || spec.Environment != facts.Environment || !slices.Contains(spec.Outputs, facts.Subject) {
		return errors.New("model recipe: dataset transform differs from candidate subject or execution evidence")
	}
	return nil
}

// CandidateDomain names the exact component domain this adapter closes.
func (codeAdmissionAdapter) CandidateDomain() string { return string(CandidateCode) }

// ValidateCandidateComponent binds the component to the candidate's code revision.
func (codeAdmissionAdapter) ValidateCandidateComponent(
	_ context.Context,
	_ artifact.Reader,
	facts runrecord.CandidateAdmissionFacts,
	component runrecord.CandidateAdmissionComponent,
) error {
	if component.Specification != facts.Code {
		return errors.New("model recipe: code component differs from candidate revision")
	}
	return nil
}
