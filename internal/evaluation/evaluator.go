package evaluation

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const (
	evaluatorVersion   uint16 = 1
	evaluatorMediaType        = "application/vnd.overgo.evaluator+json"
	evaluatorSchema           = "overgo/evaluator/v1"
)

type Evaluator struct {
	ID         artifact.ID      `json:"-"`
	Version    uint16           `json:"version"`
	Plan       artifact.ID      `json:"plan"`
	Acceptance artifact.ID      `json:"acceptance"`
	Metrics    []MetricContract `json:"metrics"`
}

type KnownOutcome struct {
	Model   artifact.ID        `json:"model"`
	Rank    uint32             `json:"rank"`
	Metrics []runrecord.Metric `json:"metrics"`
}

var evaluatorCodec = artifact.JSONDocumentCodec(
	"evaluator", artifact.KindEvidence, evaluatorMediaType, evaluatorSchema,
	func(value *Evaluator) error {
		if value == nil || value.Version != evaluatorVersion || value.Plan.Kind() != artifact.KindProfile ||
			value.Acceptance.Kind() != artifact.KindProfile || len(value.Metrics) == 0 {
			return errors.New("evaluation: invalid evaluator")
		}
		sort.Slice(value.Metrics, func(i, j int) bool { return value.Metrics[i].Name < value.Metrics[j].Name })
		for index, metric := range value.Metrics {
			if metric.Name == "" || !validMetricDirection(metric.Direction) || index > 0 && value.Metrics[index-1].Name == metric.Name {
				return errors.New("evaluation: invalid evaluator metric")
			}
		}
		return nil
	}, func(value Evaluator) artifact.ID { return value.ID },
	func(value *Evaluator, id artifact.ID) { value.ID = id },
	func(value Evaluator) Evaluator { value.Metrics = slices.Clone(value.Metrics); return value },
)

func NewEvaluator(plan artifact.ID, acceptance AcceptancePolicy) (Evaluator, error) {
	return evaluatorCodec.New(Evaluator{
		Version: evaluatorVersion, Plan: plan, Acceptance: acceptance.ID, Metrics: slices.Clone(acceptance.Metrics),
	})
}

func (value Evaluator) Content() (artifact.Content, error) { return evaluatorCodec.Content(value) }

func (value Evaluator) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Plan, value.Acceptance)
}

func ValidateEvaluatorPromotion(
	candidate Evaluator,
	outcomeSet artifact.ID,
	outcomes []KnownOutcome,
	current, prior runrecord.AdmissionBinding,
	approval recipe.Decision,
) error {
	identified, err := artifact.JSONID(artifact.KindDatasetShard, outcomes)
	if err != nil || identified != outcomeSet || current.SealedInputs != outcomeSet ||
		evaluatorCodec.ValidateIdentity(candidate) != nil || current.Evaluator.Identity != candidate.ID || approval.Subject != candidate.ID ||
		!slices.Contains(approval.Evidence, current.SealedInputs) {
		return errors.New("evaluation: evaluator promotion authority differs")
	}
	if err := runrecord.ValidateAdmissionSuccession(current, prior, approval); err != nil {
		return err
	}
	return validateKnownOutcomeOrder(candidate.Metrics, outcomes)
}

func validateKnownOutcomeOrder(contract []MetricContract, outcomes []KnownOutcome) error {
	if len(outcomes) < 2 {
		return errors.New("evaluation: known outcome set is incomplete")
	}
	for index, value := range outcomes {
		if value.Model.Kind() != artifact.KindModel || !metricsMatch(contract, value.Metrics) || index > 0 && outcomes[index-1].Rank >= value.Rank {
			return errors.New("evaluation: invalid known outcome")
		}
		if index > 0 && !dominates(outcomes[index-1].Metrics, value.Metrics) {
			return errors.New("evaluation: evaluator misranks known outcomes")
		}
	}
	return nil
}

func metricsMatch(contract []MetricContract, metrics []runrecord.Metric) bool {
	metrics = slices.Clone(metrics)
	sort.Slice(metrics, func(i, j int) bool { return metrics[i].Name < metrics[j].Name })
	if len(contract) != len(metrics) {
		return false
	}
	for index, expected := range contract {
		actual := metrics[index]
		if actual.Name != expected.Name || actual.Unit != expected.Unit || actual.Direction != expected.Direction {
			return false
		}
	}
	return true
}

func dominates(better, worse []runrecord.Metric) bool {
	better, worse = slices.Clone(better), slices.Clone(worse)
	sort.Slice(better, func(i, j int) bool { return better[i].Name < better[j].Name })
	sort.Slice(worse, func(i, j int) bool { return worse[i].Name < worse[j].Name })
	if len(better) != len(worse) {
		return false
	}
	strict := false
	for index := range better {
		left, right := better[index], worse[index]
		if left.Name != right.Name || left.Unit != right.Unit || left.Direction != right.Direction {
			return false
		}
		switch left.Direction {
		case runrecord.DirectionMaximize:
			if left.Value < right.Value {
				return false
			}
			strict = strict || left.Value > right.Value
		case runrecord.DirectionMinimize:
			if left.Value > right.Value {
				return false
			}
			strict = strict || left.Value < right.Value
		default:
			return false
		}
	}
	return strict
}
