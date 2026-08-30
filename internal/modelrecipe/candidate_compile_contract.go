package modelrecipe

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// CandidateCompileFacts are the immutable common facts visible to a domain
// plugin. Every dispatch receives an isolated copy; plugins cannot replace the
// candidate, admission, budgets, splits, falsifier, code, environment, or evidence.
type CandidateCompileFacts struct {
	Candidate         artifact.ID               `json:"candidate"`
	Admission         artifact.ID               `json:"admission"`
	Subject           artifact.ID               `json:"subject"`
	Parent            artifact.ID               `json:"parent"`
	Prediction        recipe.SteeringPrediction `json:"prediction"`
	CostUnit          string                    `json:"cost_unit"`
	Falsifier         artifact.ID               `json:"falsifier"`
	References        []CandidateReference      `json:"references"`
	DevelopmentSplit  artifact.ID               `json:"development_split"`
	PromotionSplit    artifact.ID               `json:"promotion_split"`
	DevelopmentBudget artifact.ID               `json:"development_budget"`
	PromotionBudget   artifact.ID               `json:"promotion_budget"`
	Code              artifact.ID               `json:"code"`
	Environment       artifact.ID               `json:"environment"`
}

// CandidateDomainAblation is a plugin-derived alternate realization. The
// plugin supplies facts only; the common compiler owns the persisted ablation.
type CandidateDomainAblation struct {
	Omitted     artifact.ID `json:"omitted"`
	Realization artifact.ID `json:"realization"`
}

// CandidateComponentCompilation carries a plugin's non-authorizing facts.
// Plan identity and ablation documents are always derived by the common owner.
type CandidateComponentCompilation struct {
	Plan      CandidateComponentPlan
	Ablations []CandidateDomainAblation
	Contents  []artifact.Content
	Lineage   []artifact.Lineage
}

// CandidateDomainPlugin contributes domain validation and deterministic plan
// derivation only. The compiler receives a compiled Go catalog explicitly;
// there is no filesystem discovery, transport, executor, or runtime registry.
type CandidateDomainPlugin interface {
	CandidateDomain() string
	AdmissionAdapter() runrecord.CandidateComponentAdmissionAdapter
	EvaluatorPlugin() CandidateEvaluatorPlugin
	CompileCandidateComponent(
		context.Context,
		artifact.Reader,
		CandidateCompileFacts,
		CandidateComponent,
	) (CandidateComponentCompilation, error)
}
