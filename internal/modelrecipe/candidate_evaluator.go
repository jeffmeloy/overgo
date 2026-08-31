package modelrecipe

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
)

const (
	// CandidateEvaluationIntentMediaType identifies the typed pre-execution falsifier.
	CandidateEvaluationIntentMediaType = "application/vnd.overgo.candidate-evaluation-intent+json"
	// CandidateEvaluationIntentSchema identifies the evaluator intent contract.
	CandidateEvaluationIntentSchema = "overgo/candidate-evaluation-intent/v1"
)

// CandidateStopRule names a compiled fail-closed evaluation stop condition.
type CandidateStopRule string

const (
	// CandidateStopOnBudgetOrNonPositiveIsolatedImprovement stops when either
	// split budget is exhausted or isolated held-out improvement is non-positive.
	CandidateStopOnBudgetOrNonPositiveIsolatedImprovement CandidateStopRule = "budget-exhausted-or-non-positive-isolated-improvement"
)

// CandidateEvaluationIntent binds a candidate hypothesis to its exact metric,
// cost unit, and fail-closed stop rule before any domain realization runs.
type CandidateEvaluationIntent struct {
	Version  uint16            `json:"version"`
	Subject  artifact.ID       `json:"subject"`
	Metric   string            `json:"metric"`
	CostUnit string            `json:"cost_unit"`
	StopRule CandidateStopRule `json:"stop_rule"`
	ID       artifact.ID       `json:"-"`
}

// CandidateEvaluatorPlugin validates one compiled-Go evaluator contract.
// Evaluation execution remains owned by the later evaluation lifecycle.
type CandidateEvaluatorPlugin interface {
	CandidateEvaluator() string
	ValidateCandidateEvaluation(context.Context, artifact.Reader, CandidateCompileFacts) error
}

var candidateEvaluationIntentCodec = artifact.JSONDocumentCodec(
	"candidate evaluation intent", artifact.KindRecipe,
	CandidateEvaluationIntentMediaType, CandidateEvaluationIntentSchema,
	canonicalizeCandidateEvaluationIntent,
	func(value CandidateEvaluationIntent) artifact.ID { return value.ID },
	func(value *CandidateEvaluationIntent, id artifact.ID) { value.ID = id }, nil,
)

// NewCandidateEvaluationIntent canonicalizes one typed fail-closed evaluator intent.
func NewCandidateEvaluationIntent(
	subject artifact.ID,
	metric, costUnit string,
	stopRule CandidateStopRule,
) (CandidateEvaluationIntent, error) {
	return candidateEvaluationIntentCodec.NewInitial(CandidateEvaluationIntent{
		Subject: subject,
		Metric:  metric, CostUnit: costUnit, StopRule: stopRule,
	})
}

// Content returns the immutable evaluator intent.
func (value CandidateEvaluationIntent) Content() (artifact.Content, error) {
	return candidateEvaluationIntentCodec.Content(value)
}

// Lineage binds the evaluator intent to the exact admitted hypothesis.
func (value CandidateEvaluationIntent) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Subject)
}

// RequireCandidateEvaluationIntent replays the exact typed falsifier declared
// by one candidate before that candidate can consume realization budget.
func RequireCandidateEvaluationIntent(
	ctx context.Context,
	reader artifact.Reader,
	candidate Candidate,
) (CandidateEvaluationIntent, error) {
	if err := candidate.ValidateIdentity(); err != nil {
		return CandidateEvaluationIntent{}, err
	}
	spec := candidate.Spec()
	return requireCandidateEvaluationIntent(
		ctx, reader, spec.Falsifier, spec.Subject, spec.Prediction.Metric, spec.CostUnit,
	)
}

// CandidateEvaluationIntentValidator checks that an admitted falsifier is the
// exact typed evaluation intent compiled for the candidate authority.
type CandidateEvaluationIntentValidator struct{}

// CandidateEvaluator identifies the evaluator contract accepted by this validator.
func (CandidateEvaluationIntentValidator) CandidateEvaluator() string {
	return CandidateEvaluationIntentSchema
}

// ValidateCandidateEvaluation checks exact identity, lineage, metric, and cost authority.
func (CandidateEvaluationIntentValidator) ValidateCandidateEvaluation(
	ctx context.Context,
	reader artifact.Reader,
	facts CandidateCompileFacts,
) error {
	_, err := requireCandidateEvaluationIntent(
		ctx, reader, facts.Falsifier, facts.Subject, facts.Prediction.Metric, facts.CostUnit,
	)
	return err
}

func requireCandidateEvaluationIntent(
	ctx context.Context,
	reader artifact.Reader,
	id, subject artifact.ID,
	metric, costUnit string,
) (CandidateEvaluationIntent, error) {
	intent, err := candidateEvaluationIntentCodec.RequireExactLineage(
		ctx, reader, id, CandidateEvaluationIntent.Lineage,
	)
	if err != nil {
		return CandidateEvaluationIntent{}, err
	}
	expected, err := NewCandidateEvaluationIntent(
		subject, metric, costUnit, CandidateStopOnBudgetOrNonPositiveIsolatedImprovement,
	)
	if err != nil {
		return CandidateEvaluationIntent{}, err
	}
	if intent.ID != expected.ID || intent.Subject != subject ||
		intent.Metric != metric || intent.CostUnit != costUnit {
		return CandidateEvaluationIntent{}, errors.New("model recipe: evaluation intent differs from candidate authority")
	}
	return intent, nil
}

func canonicalizeCandidateEvaluationIntent(value *CandidateEvaluationIntent) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !value.Subject.Valid() ||
		strings.TrimSpace(value.Metric) == "" || value.Metric != strings.TrimSpace(value.Metric) ||
		strings.TrimSpace(value.CostUnit) == "" || value.CostUnit != strings.TrimSpace(value.CostUnit) ||
		value.StopRule != CandidateStopOnBudgetOrNonPositiveIsolatedImprovement {
		return errors.New("model recipe: invalid candidate evaluation intent")
	}
	return nil
}
