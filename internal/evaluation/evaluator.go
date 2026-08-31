package evaluation

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
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

type KnownOutcome struct {
	Model   artifact.ID        `json:"model"`
	Rank    uint32             `json:"rank"`
	Metrics []runrecord.Metric `json:"metrics"`
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
	func(value Evaluator) Evaluator { value.Metrics = slices.Clone(value.Metrics); return value },
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
	var previous []runrecord.Metric
	for index, value := range outcomes {
		metrics := slices.Clone(value.Metrics)
		slices.SortFunc(metrics, func(left, right runrecord.Metric) int { return cmp.Compare(left.Name, right.Name) })
		if value.Model.Kind() != artifact.KindModel || len(contract) != len(metrics) || !orderedMetricContractAdmits(contract, metrics) ||
			index > 0 && outcomes[index-1].Rank >= value.Rank {
			return errors.New("evaluation: invalid known outcome")
		}
		if index > 0 && !orderedMetricsDominate(previous, metrics) {
			return errors.New("evaluation: evaluator misranks known outcomes")
		}
		previous = metrics
	}
	return nil
}

func orderedMetricsDominate(better, worse []runrecord.Metric) bool {
	noRegression, strict := orderedMetricRelation(better, worse)
	return noRegression && strict
}

// orderedMetricRelation compares two already-canonical metric vectors without
// scalarizing distinct quality dimensions.
func orderedMetricRelation(candidate, baseline []runrecord.Metric) (noRegression, strict bool) {
	if len(candidate) == 0 || len(candidate) != len(baseline) {
		return false, false
	}
	var neutral float64
	for index := range candidate {
		left, right := candidate[index], baseline[index]
		if left.Name != right.Name || left.Unit != right.Unit || left.Direction != right.Direction {
			return false, false
		}
		advantage, ok := left.Direction.Advantage(left.Value, right.Value)
		if !ok || advantage < neutral {
			return false, false
		}
		strict = strict || advantage > neutral
	}
	return true, strict
}
