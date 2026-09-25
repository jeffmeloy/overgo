package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

const (
	heldoutVerdictMediaType = "application/vnd.overgo.heldout-verdict+json"
	heldoutVerdictSchema    = "overgo/heldout-verdict/v1"
)

var heldoutVerdictCodec = artifact.JSONDocumentCodec(
	"held-out verdict", artifact.KindEvaluation, heldoutVerdictMediaType, heldoutVerdictSchema,
	canonicalizeHeldoutVerdict, func(value HeldoutVerdict) artifact.ID { return value.ID },
	func(value *HeldoutVerdict, id artifact.ID) { value.ID = id }, func(value HeldoutVerdict) HeldoutVerdict { return value },
)

// heldoutNumericScorers maps an objective's declared metric to the numeric
// scorer that measures it on held-out records.
var heldoutNumericScorers = map[trainingprogram.EvaluationMetric]NumericScorer{
	trainingprogram.MetricForecastMAE:   NumericAbsoluteError,
	trainingprogram.MetricTableAccuracy: NumericLabelAccuracy,
}

// HeldoutVerdict judges a trained model against its base model on one
// objective's held-out view: both are scored by the same numeric target plan,
// and the verdict passes when the trained model strictly improves the metric
// the objective declares, in the metric's own direction. No threshold is
// involved; Passed is recomputed from the recorded values.
type HeldoutVerdict struct {
	ID             artifact.ID                      `json:"-"`
	Version        uint16                           `json:"version"`
	Objective      artifact.ID                      `json:"objective"`
	View           artifact.ID                      `json:"view"`
	Plan           artifact.ID                      `json:"plan"`
	Observation    artifact.ID                      `json:"observation"`
	Baseline       artifact.ID                      `json:"baseline"`
	Candidate      artifact.ID                      `json:"candidate"`
	Metric         trainingprogram.EvaluationMetric `json:"metric"`
	Direction      runrecord.Direction              `json:"direction"`
	BaselineValue  float64                          `json:"baseline_value"`
	CandidateValue float64                          `json:"candidate_value"`
	Passed         bool                             `json:"passed"`
}

// JudgeHeldout compares the base model's and the trained model's numeric
// target reports on one plan bound to the objective's held-out view. The
// observation is the training session that produced the trained model.
func JudgeHeldout(
	objective trainingprogram.ObjectiveDocument,
	view SFTEvaluationView,
	plan NumericTargetPlan,
	baseline, candidate NumericTargetReport,
	observation artifact.ID,
) (HeldoutVerdict, error) {
	if _, err := objective.Content(); err != nil {
		return HeldoutVerdict{}, err
	}
	if err := sftEvaluationViewCodec.ValidateIdentity(view); err != nil {
		return HeldoutVerdict{}, err
	}
	if err := numericTargetPlanCodec.ValidateIdentity(plan); err != nil {
		return HeldoutVerdict{}, err
	}
	for _, report := range []NumericTargetReport{baseline, candidate} {
		if err := numericTargetReportCodec.ValidateIdentity(report); err != nil {
			return HeldoutVerdict{}, err
		}
	}
	if view.Objective != objective.ID || plan.View != view.ID || baseline.Plan != plan.ID || candidate.Plan != plan.ID {
		return HeldoutVerdict{}, errors.New("evaluation: held-out reports are not bound to this objective's view")
	}
	scorer, found := heldoutNumericScorers[objective.Metric]
	if !found {
		return HeldoutVerdict{}, fmt.Errorf("evaluation: no held-out numeric scorer measures %q", objective.Metric)
	}
	baseMetric, baseFound := reportMetric(baseline, scorer)
	candidateMetric, candidateFound := reportMetric(candidate, scorer)
	if !baseFound || !candidateFound || baseMetric.Direction != candidateMetric.Direction {
		return HeldoutVerdict{}, fmt.Errorf("evaluation: held-out reports do not both carry %q", scorer)
	}
	return heldoutVerdictCodec.NewInitial(HeldoutVerdict{
		Objective: objective.ID, View: view.ID, Plan: plan.ID, Observation: observation,
		Baseline: baseline.ID, Candidate: candidate.ID, Metric: objective.Metric,
		Direction: baseMetric.Direction, BaselineValue: baseMetric.Value, CandidateValue: candidateMetric.Value,
	})
}

// Content encodes the verdict as its canonical artifact document.
func (verdict HeldoutVerdict) Content() (artifact.Content, error) {
	return heldoutVerdictCodec.Content(verdict)
}

// Batch stages the verdict with lineage to the objective, view, plan,
// training observation and both reports it judged.
func (verdict HeldoutVerdict) Batch(key string) (artifact.Batch, error) {
	return heldoutVerdictCodec.Batch(key, verdict, artifact.DependencyLineage(
		verdict.ID, verdict.Objective, verdict.View, verdict.Plan, verdict.Observation, verdict.Baseline, verdict.Candidate,
	), nil)
}

// CommittedHeldoutVerdictPassed reads one committed held-out verdict and
// reports whether it passed for current, re-deriving everything it claims:
// the objective it judged declares current's contract, its view recompiles
// from the committed training and held-out memberships, its reports recompute
// to the same verdict, and its training observation is evidence current
// already carries.
func CommittedHeldoutVerdictPassed(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	current trainingprogram.ObjectiveDocument,
) (bool, error) {
	verdict, err := heldoutVerdictCodec.Require(ctx, reader, id)
	if err != nil {
		return false, err
	}
	objective, err := trainingprogram.LoadObjective(ctx, reader, verdict.Objective)
	if err != nil {
		return false, err
	}
	if !trainingprogram.SameContract(objective, current) {
		return false, fmt.Errorf("evaluation: verdict %s judged a different objective contract", id)
	}
	view, err := sftEvaluationViewCodec.Require(ctx, reader, verdict.View)
	if err != nil {
		return false, err
	}
	training, heldout, err := viewMemberships(ctx, reader, view)
	if err != nil {
		return false, err
	}
	recompiled, err := CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		return false, err
	}
	if recompiled.ID != view.ID {
		return false, fmt.Errorf("evaluation: verdict %s view does not recompile from its memberships", id)
	}
	plan, err := numericTargetPlanCodec.Require(ctx, reader, verdict.Plan)
	if err != nil {
		return false, err
	}
	baseline, err := numericTargetReportCodec.Require(ctx, reader, verdict.Baseline)
	if err != nil {
		return false, err
	}
	candidate, err := numericTargetReportCodec.Require(ctx, reader, verdict.Candidate)
	if err != nil {
		return false, err
	}
	judged, err := JudgeHeldout(objective, view, plan, baseline, candidate, verdict.Observation)
	if err != nil {
		return false, err
	}
	if judged.ID != verdict.ID {
		return false, fmt.Errorf("evaluation: verdict %s differs from its recomputation", id)
	}
	if !slices.Contains(current.Evidence, verdict.Observation) {
		return false, fmt.Errorf("evaluation: verdict %s training observation is not evidence of this objective", id)
	}
	return verdict.Passed, nil
}

// IsHeldoutVerdict reports whether a descriptor names a held-out verdict.
func IsHeldoutVerdict(descriptor artifact.Descriptor) bool {
	return descriptor.MediaType == heldoutVerdictMediaType && descriptor.Schema == heldoutVerdictSchema
}

func viewMemberships(ctx context.Context, reader artifact.Reader, view SFTEvaluationView) (dataset.Membership, dataset.Membership, error) {
	training, trainingFound, err := dataset.LoadMembership(ctx, reader, view.TrainingMembership)
	if err != nil {
		return dataset.Membership{}, dataset.Membership{}, err
	}
	heldout, heldoutFound, err := dataset.LoadMembership(ctx, reader, view.HeldoutMembership)
	if err != nil {
		return dataset.Membership{}, dataset.Membership{}, err
	}
	if !trainingFound || !heldoutFound {
		return dataset.Membership{}, dataset.Membership{}, errors.New("evaluation: held-out view memberships are not committed")
	}
	return training, heldout, nil
}

func reportMetric(report NumericTargetReport, scorer NumericScorer) (runrecord.Metric, bool) {
	index := slices.IndexFunc(report.Metrics, func(metric runrecord.Metric) bool { return metric.Name == string(scorer) })
	if index < 0 {
		return runrecord.Metric{}, false
	}
	return report.Metrics[index], true
}

func canonicalizeHeldoutVerdict(verdict *HeldoutVerdict) error {
	if verdict == nil || verdict.Version != artifact.InitialDocumentVersion ||
		verdict.Objective.Kind() != artifact.KindProfile || verdict.View.Kind() != artifact.KindProfile ||
		verdict.Plan.Kind() != artifact.KindProfile || verdict.Observation.Kind() != artifact.KindEvidence ||
		verdict.Baseline.Kind() != artifact.KindEvaluation || verdict.Candidate.Kind() != artifact.KindEvaluation ||
		verdict.Baseline == verdict.Candidate || verdict.Metric == "" || !validMetricDirection(verdict.Direction) {
		return errors.New("evaluation: invalid held-out verdict")
	}
	switch verdict.Direction {
	case runrecord.DirectionMinimize:
		verdict.Passed = verdict.CandidateValue < verdict.BaselineValue
	default:
		verdict.Passed = verdict.CandidateValue > verdict.BaselineValue
	}
	return nil
}
