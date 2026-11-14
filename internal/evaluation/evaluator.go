package evaluation

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	evaluatorMediaType = "application/vnd.overgo.evaluator+json"
	evaluatorSchema    = "overgo/evaluator/v1"
)

type Evaluator struct {
	ID         artifact.ID      `json:"-"`
	Version    uint16           `json:"version"`
	Plan       artifact.ID      `json:"plan"`
	Acceptance artifact.ID      `json:"acceptance"`
	Metrics    []MetricContract `json:"metrics"`
}

var evaluatorCodec = artifact.JSONDocumentCodec(
	"evaluator", artifact.KindEvidence, evaluatorMediaType, evaluatorSchema,
	func(value *Evaluator) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Plan.Kind() != artifact.KindProfile ||
			value.Acceptance.Kind() != artifact.KindProfile || len(value.Metrics) == 0 {
			return errors.New("evaluation: invalid evaluator")
		}
		if !canonicalizeMetricContracts(value.Metrics) {
			return errors.New("evaluation: invalid evaluator metric")
		}
		return nil
	}, func(value Evaluator) artifact.ID { return value.ID },
	func(value *Evaluator, id artifact.ID) { value.ID = id },
)

func NewEvaluator(plan artifact.ID, acceptance AcceptancePolicy) (Evaluator, error) {
	return evaluatorCodec.NewInitial(Evaluator{
		Plan: plan, Acceptance: acceptance.ID, Metrics: slices.Clone(acceptance.Metrics),
	})
}

func (value Evaluator) Content() (artifact.Content, error) { return evaluatorCodec.Content(value) }

func (value Evaluator) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Plan, value.Acceptance)
}
