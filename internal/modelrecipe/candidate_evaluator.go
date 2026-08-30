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
	return candidateEvaluationIntentCodec.NewPrepared(CandidateEvaluationIntent{
		Subject: subject,
		Metric:  metric, CostUnit: costUnit, StopRule: stopRule,
	}, func(value *CandidateEvaluationIntent) {
		value.Version = artifact.InitialDocumentVersion
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
	intent, err := candidateEvaluationIntentCodec.RequireExactLineage(
		ctx, reader, facts.Falsifier, CandidateEvaluationIntent.Lineage,
	)
	if err != nil {
		return err
	}
	expected, err := NewCandidateEvaluationIntent(
		facts.Subject, facts.Prediction.Metric, facts.CostUnit,
		CandidateStopOnBudgetOrNonPositiveIsolatedImprovement,
	)
	if err != nil {
		return err
	}
	if intent.ID != expected.ID || intent.Subject != facts.Subject ||
		intent.Metric != facts.Prediction.Metric || intent.CostUnit != facts.CostUnit {
		return errors.New("model recipe: evaluation intent differs from candidate authority")
	}
	return nil
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
