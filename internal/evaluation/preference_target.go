package evaluation

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/trainingprogram"
)

const (
	preferenceTargetPlanMedia    = "application/vnd.overgo.preference-target-plan+json"
	preferenceTargetPlanSchema   = "overgo/preference-target-plan/v1"
	preferenceTargetReportMedia  = "application/vnd.overgo.preference-target-report+json"
	preferenceTargetReportSchema = "overgo/preference-target-report/v1"

	preferenceAccuracyMetric = "chosen-vs-rejected-accuracy"
	preferenceLossMetric     = "dpo-loss"
	preferenceMarginMetric   = "policy-reference-margin"
)

type PreferenceTargetSuite struct {
	Policy trainingprogram.PreferencePolicy `json:"policy"`
}

type PreferenceTargetPlan struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	View      artifact.ID                      `json:"view"`
	Objective artifact.ID                      `json:"objective"`
	Heldout   artifact.ID                      `json:"heldout"`
	Signature recipecontract.ModalitySignature `json:"signature"`
	Policy    trainingprogram.PreferencePolicy `json:"policy"`
	Records   []string                         `json:"records"`
}

type PreferenceTargetObservation struct {
	Record            string              `json:"record"`
	PolicyChosen      sequencescore.Score `json:"policy_chosen"`
	PolicyRejected    sequencescore.Score `json:"policy_rejected"`
	ReferenceChosen   sequencescore.Score `json:"reference_chosen"`
	ReferenceRejected sequencescore.Score `json:"reference_rejected"`
}

type PreferenceRecordResult struct {
	PreferenceTargetObservation
	PolicyMargin    float64 `json:"policy_margin"`
	ReferenceMargin float64 `json:"reference_margin"`
	RelativeMargin  float64 `json:"relative_margin"`
	Chosen          bool    `json:"chosen"`
	Loss            float64 `json:"loss"`
}

type PreferenceTargetReport struct {
	ID      artifact.ID              `json:"-"`
	Version uint16                   `json:"version"`
	Plan    artifact.ID              `json:"plan"`
	Records []PreferenceRecordResult `json:"records"`
	Metrics []runrecord.Metric       `json:"metrics"`
}

var (
	preferenceTargetPlanCodec = artifact.JSONDocumentCodec(
		"preference target plan", artifact.KindProfile, preferenceTargetPlanMedia, preferenceTargetPlanSchema,
		canonicalizePreferenceTargetPlan, func(value PreferenceTargetPlan) artifact.ID { return value.ID },
		func(value *PreferenceTargetPlan, id artifact.ID) { value.ID = id }, clonePreferenceTargetPlan,
	)
	preferenceTargetReportCodec = artifact.JSONDocumentCodec(
		"preference target report", artifact.KindEvaluation, preferenceTargetReportMedia, preferenceTargetReportSchema,
		canonicalizePreferenceTargetReport, func(value PreferenceTargetReport) artifact.ID { return value.ID },
		func(value *PreferenceTargetReport, id artifact.ID) { value.ID = id }, clonePreferenceTargetReport,
	)
)

// CompilePreferenceTargetPlan binds preference evaluation authorities.
func CompilePreferenceTargetPlan(
	objective trainingprogram.ObjectiveDocument,
	view SFTEvaluationView,
	suite PreferenceTargetSuite,
) (PreferenceTargetPlan, error) {
	if _, err := objective.Content(); err != nil {
		return PreferenceTargetPlan{}, err
	}
	if _, err := view.Content(); err != nil {
		return PreferenceTargetPlan{}, err
	}
	if objective.Kind != trainingprogram.ObjectiveDPO || objective.ID != view.Objective || suite.Policy.Validate() != nil {
		return PreferenceTargetPlan{}, errors.New("evaluation: preference authority differs from DPO objective")
	}
	records := make([]string, len(view.Records))
	for index, record := range view.Records {
		records[index] = record.ID
	}
	return preferenceTargetPlanCodec.NewInitial(PreferenceTargetPlan{
		View: view.ID, Objective: objective.ID,
		Heldout: view.HeldoutMembership, Signature: view.Signature.Clone(), Policy: suite.Policy, Records: records,
	})
}

func (plan PreferenceTargetPlan) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{plan.View, plan.Objective, plan.Heldout, plan.Policy.Reference}
	return preferenceTargetPlanCodec.Batch(key, plan, artifact.DependencyLineage(plan.ID, parents...), nil)
}

// ScorePreferenceTargets evaluates preference observations against frozen authorities.
func ScorePreferenceTargets(plan PreferenceTargetPlan, observations []PreferenceTargetObservation) (PreferenceTargetReport, error) {
	if err := preferenceTargetPlanCodec.ValidateIdentity(plan); err != nil || len(observations) != len(plan.Records) {
		return PreferenceTargetReport{}, errors.New("evaluation: preference observations differ from plan")
	}
	observations = slices.Clone(observations)
	sort.Slice(observations, func(i, j int) bool { return observations[i].Record < observations[j].Record })
	report := PreferenceTargetReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.ID, Records: make([]PreferenceRecordResult, len(plan.Records)),
	}
	var accuracy, loss, relativeMargin float64
	for index, observation := range observations {
		if observation.Record != plan.Records[index] {
			return PreferenceTargetReport{}, errors.New("evaluation: preference record differs from plan")
		}
		scores := trainingprogram.PreferenceScores{
			PolicyChosen: observation.PolicyChosen, PolicyRejected: observation.PolicyRejected,
			ReferenceChosen: observation.ReferenceChosen, ReferenceRejected: observation.ReferenceRejected,
		}
		result, err := trainingprogram.DPOLoss(scores, plan.Policy.Scale)
		if err != nil {
			return PreferenceTargetReport{}, err
		}
		policyMargin := observation.PolicyChosen.LogProbability - observation.PolicyRejected.LogProbability
		referenceMargin := observation.ReferenceChosen.LogProbability - observation.ReferenceRejected.LogProbability
		chosen := policyMargin > 0
		if chosen {
			accuracy++
		}
		loss += result.Loss
		relativeMargin += policyMargin - referenceMargin
		report.Records[index] = PreferenceRecordResult{
			PreferenceTargetObservation: observation, PolicyMargin: policyMargin, ReferenceMargin: referenceMargin,
			RelativeMargin: policyMargin - referenceMargin, Chosen: chosen, Loss: result.Loss,
		}
	}
	count := float64(len(report.Records))
	report.Metrics = []runrecord.Metric{
		{Name: preferenceAccuracyMetric, Value: accuracy / count, Direction: runrecord.DirectionMaximize},
		{Name: preferenceLossMetric, Value: loss / count, Direction: runrecord.DirectionMinimize},
		{Name: preferenceMarginMetric, Value: relativeMargin / count, Direction: runrecord.DirectionMaximize},
	}
	return preferenceTargetReportCodec.New(report)
}

func (report PreferenceTargetReport) Batch(key string) (artifact.Batch, error) {
	return preferenceTargetReportCodec.Batch(key, report, artifact.DependencyLineage(report.ID, report.Plan), nil)
}

func canonicalizePreferenceTargetPlan(plan *PreferenceTargetPlan) error {
	if plan == nil || plan.Version != artifact.InitialDocumentVersion || plan.View.Kind() != artifact.KindProfile ||
		plan.Objective.Kind() != artifact.KindProfile || plan.Heldout.Kind() != artifact.KindDatasetShard ||
		plan.Policy.Validate() != nil || plan.Signature.Validate() != nil || len(plan.Signature.Outputs) != 1 ||
		plan.Signature.Outputs[0] != recipecontract.ModalityText || len(plan.Records) == 0 {
		return errors.New("evaluation: invalid preference target plan")
	}
	slices.Sort(plan.Records)
	for index, record := range plan.Records {
		if record == "" || index > 0 && plan.Records[index-1] == record {
			return errors.New("evaluation: invalid preference target record")
		}
	}
	return nil
}

func canonicalizePreferenceTargetReport(report *PreferenceTargetReport) error {
	if report == nil || report.Version != artifact.InitialDocumentVersion || report.Plan.Kind() != artifact.KindProfile ||
		len(report.Records) == 0 || len(report.Metrics) != 3 {
		return errors.New("evaluation: invalid preference target report")
	}
	for index, record := range report.Records {
		if record.Record == "" || index > 0 && report.Records[index-1].Record >= record.Record ||
			!finite(record.PolicyMargin) || !finite(record.ReferenceMargin) || !finite(record.RelativeMargin) ||
			!finite(record.Loss) || record.RelativeMargin != record.PolicyMargin-record.ReferenceMargin {
			return errors.New("evaluation: invalid preference target result")
		}
	}
	for index, name := range []string{preferenceAccuracyMetric, preferenceLossMetric, preferenceMarginMetric} {
		if report.Metrics[index].Name != name || !finite(report.Metrics[index].Value) {
			return errors.New("evaluation: invalid preference target metric")
		}
	}
	if report.Metrics[0].Direction != runrecord.DirectionMaximize || report.Metrics[1].Direction != runrecord.DirectionMinimize ||
		report.Metrics[2].Direction != runrecord.DirectionMaximize {
		return errors.New("evaluation: invalid preference metric direction")
	}
	return nil
}

func clonePreferenceTargetPlan(plan PreferenceTargetPlan) PreferenceTargetPlan {
	plan.Signature = plan.Signature.Clone()
	plan.Records = slices.Clone(plan.Records)
	return plan
}

func clonePreferenceTargetReport(report PreferenceTargetReport) PreferenceTargetReport {
	report.Records = slices.Clone(report.Records)
	report.Metrics = slices.Clone(report.Metrics)
	return report
}
