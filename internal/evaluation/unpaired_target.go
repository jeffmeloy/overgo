package evaluation

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/trainingprogram"
)

const (
	unpairedTargetPlanMedia    = "application/vnd.overgo.unpaired-preference-target-plan+json"
	unpairedTargetPlanSchema   = "overgo/unpaired-preference-target-plan/v1"
	unpairedTargetReportMedia  = "application/vnd.overgo.unpaired-preference-target-report+json"
	unpairedTargetReportSchema = "overgo/unpaired-preference-target-report/v1"

	unpairedLossMetric   = "kto-loss"
	unpairedMarginMetric = "kto-reward-margin"
)

// unpairedTargetRecord is one held-out unpaired completion and its label.
type unpairedTargetRecord struct {
	ID        string `json:"id"`
	Desirable bool   `json:"desirable"`
}

// unpairedTargetPlan binds a KTO objective, its held-out view, the KTO
// policy its records are scored under, and each held-out record's label.
type unpairedTargetPlan struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	View      artifact.ID                      `json:"view"`
	Objective artifact.ID                      `json:"objective"`
	Heldout   artifact.ID                      `json:"heldout"`
	Signature recipecontract.ModalitySignature `json:"signature"`
	Policy    trainingprogram.PreferencePolicy `json:"policy"`
	Records   []unpairedTargetRecord           `json:"records"`
}

// unpairedTargetObservation carries one record's label, its completion
// scored after its own prompt and its partner's completion scored after
// that prompt, each under the policy and the reference.
type unpairedTargetObservation struct {
	Record              string              `json:"record"`
	Partner             string              `json:"partner"`
	Desirable           bool                `json:"desirable"`
	Policy              sequencescore.Score `json:"policy"`
	Reference           sequencescore.Score `json:"reference"`
	MismatchedPolicy    sequencescore.Score `json:"mismatched_policy"`
	MismatchedReference sequencescore.Score `json:"mismatched_reference"`
}

// unpairedRecordResult adds the record's implied reward r to its
// observation.
type unpairedRecordResult struct {
	unpairedTargetObservation
	Reward float64 `json:"reward"`
}

// unpairedMetrics are a report's metrics in order: the mean KTO loss and
// the desirable-minus-undesirable reward margin.
var unpairedMetrics = []struct {
	name      string
	direction runrecord.Direction
}{{unpairedLossMetric, runrecord.DirectionMinimize}, {unpairedMarginMetric, runrecord.DirectionMaximize}}

// unpairedTargetReport holds per-record results, the view's KL estimate,
// and the mean KTO loss and desirable-minus-undesirable reward margin.
type unpairedTargetReport struct {
	ID      artifact.ID            `json:"-"`
	Version uint16                 `json:"version"`
	Plan    artifact.ID            `json:"plan"`
	KL      float64                `json:"kl"`
	Records []unpairedRecordResult `json:"records"`
	Metrics []runrecord.Metric     `json:"metrics"`
}

var (
	unpairedTargetPlanCodec = artifact.JSONDocumentCodec(
		"unpaired preference target plan", artifact.KindProfile, unpairedTargetPlanMedia, unpairedTargetPlanSchema,
		canonicalizeUnpairedTargetPlan, func(value unpairedTargetPlan) artifact.ID { return value.ID },
		func(value *unpairedTargetPlan, id artifact.ID) { value.ID = id }, cloneUnpairedTargetPlan,
	)
	unpairedTargetReportCodec = artifact.JSONDocumentCodec(
		"unpaired preference target report", artifact.KindEvaluation, unpairedTargetReportMedia, unpairedTargetReportSchema,
		canonicalizeUnpairedTargetReport, func(value unpairedTargetReport) artifact.ID { return value.ID },
		func(value *unpairedTargetReport, id artifact.ID) { value.ID = id }, cloneUnpairedTargetReport,
	)
)

// CompileUnpairedHeldoutPlan is the unpaired preference held-out slice: a
// DPO objective's held-out completions, labelled desirable or not by
// labels, each scored by a model as policy against the frozen reference
// under KTO's loss at scale, with the class weights the held-out labels
// derive. Its KL estimate pairs each record's prompt with the completion
// of the record after it, cyclically in record order.
func CompileUnpairedHeldoutPlan(objective trainingprogram.ObjectiveDocument, view SFTEvaluationView, reference artifact.ID, scale float64, labels map[string]bool) (HeldoutPlan, error) {
	if _, err := objective.Content(); err != nil {
		return HeldoutPlan{}, err
	}
	if _, err := view.Content(); err != nil {
		return HeldoutPlan{}, err
	}
	if objective.Kind != trainingprogram.ObjectiveDPO || objective.Metric != trainingprogram.MetricUnpairedPreferenceLoss || objective.ID != view.Objective {
		return HeldoutPlan{}, errors.New("evaluation: unpaired preference authority differs from the objective")
	}
	records := make([]unpairedTargetRecord, len(view.Records))
	var desirable int
	for index, record := range view.Records {
		label, found := labels[record.ID]
		if !found {
			return HeldoutPlan{}, errors.New("evaluation: held-out unpaired record has no label")
		}
		if label {
			desirable++
		}
		records[index] = unpairedTargetRecord{ID: record.ID, Desirable: label}
	}
	policy, err := trainingprogram.KTOPolicy(reference, scale, desirable, len(records)-desirable)
	if err != nil {
		return HeldoutPlan{}, err
	}
	plan, err := unpairedTargetPlanCodec.NewInitial(unpairedTargetPlan{
		View: view.ID, Objective: objective.ID, Heldout: view.HeldoutMembership,
		Signature: view.Signature.Clone(), Policy: policy, Records: records,
	})
	if err != nil {
		return HeldoutPlan{}, err
	}
	return HeldoutPlan{id: plan.ID, view: plan.View, batch: plan.Batch, score: func(observations []HeldoutObservation) (HeldoutReport, error) {
		scored := make([]unpairedTargetObservation, len(observations))
		for index, observation := range observations {
			scores := observation.Unpaired
			scored[index] = unpairedTargetObservation{
				Record: observation.Record, Partner: observation.Partner, Desirable: scores.Desirable, Policy: scores.Policy,
				Reference: scores.Reference, MismatchedPolicy: scores.MismatchedPolicy, MismatchedReference: scores.MismatchedReference,
			}
		}
		report, err := scoreUnpairedTargets(plan, scored)
		if err != nil {
			return HeldoutReport{}, err
		}
		return report.heldout(), nil
	}}, nil
}

// Batch stages the plan with lineage to its view, objective, held-out
// membership, and reference.
func (plan unpairedTargetPlan) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{plan.View, plan.Objective, plan.Heldout, plan.Policy.Reference}
	return unpairedTargetPlanCodec.Batch(key, plan, artifact.DependencyLineage(plan.ID, parents...), nil)
}

// scoreUnpairedTargets scores one model's observations, one per plan
// record, each carrying its record's label and naming the next record in
// plan order as its partner.
func scoreUnpairedTargets(plan unpairedTargetPlan, observations []unpairedTargetObservation) (unpairedTargetReport, error) {
	if err := unpairedTargetPlanCodec.ValidateIdentity(plan); err != nil || len(observations) != len(plan.Records) {
		return unpairedTargetReport{}, errors.New("evaluation: unpaired observations differ from plan")
	}
	observations = slices.Clone(observations)
	slices.SortFunc(observations, func(a, b unpairedTargetObservation) int { return cmp.Compare(a.Record, b.Record) })
	scores := make([]trainingprogram.UnpairedScores, len(observations))
	report := unpairedTargetReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.ID, Records: make([]unpairedRecordResult, len(observations)),
	}
	for index, observation := range observations {
		record := plan.Records[index]
		if observation.Record != record.ID || observation.Desirable != record.Desirable ||
			observation.Partner != plan.Records[(index+1)%len(plan.Records)].ID {
			return unpairedTargetReport{}, errors.New("evaluation: unpaired record, label or partner differs from plan")
		}
		scores[index] = trainingprogram.UnpairedScores{
			Desirable: record.Desirable, Policy: observation.Policy, Reference: observation.Reference,
			MismatchedPolicy: observation.MismatchedPolicy, MismatchedReference: observation.MismatchedReference,
		}
		report.Records[index] = unpairedRecordResult{
			unpairedTargetObservation: observation, Reward: observation.Policy.LogProbability - observation.Reference.LogProbability,
		}
	}
	result, err := plan.Policy.UnpairedLoss(scores)
	if err != nil {
		return unpairedTargetReport{}, err
	}
	report.KL = result.KL
	for index, value := range []float64{result.Loss, result.DesirableReward - result.UndesirableReward} {
		report.Metrics = append(report.Metrics, runrecord.Metric{Name: unpairedMetrics[index].name, Value: value, Direction: unpairedMetrics[index].direction})
	}
	return unpairedTargetReportCodec.New(report)
}

// Batch stages the report with lineage to its plan.
func (report unpairedTargetReport) Batch(key string) (artifact.Batch, error) {
	return unpairedTargetReportCodec.Batch(key, report, artifact.DependencyLineage(report.ID, report.Plan), nil)
}

// heldout is the report as the held-out judge reads it.
func (report unpairedTargetReport) heldout() HeldoutReport {
	return HeldoutReport{id: report.ID, plan: report.Plan, metrics: report.Metrics, batch: report.Batch}
}

// rescore scores the sequence scores each of report's records retains under
// plan.
func (plan unpairedTargetPlan) rescore(report unpairedTargetReport) (HeldoutReport, error) {
	observations := make([]unpairedTargetObservation, len(report.Records))
	for index, record := range report.Records {
		observations[index] = record.unpairedTargetObservation
	}
	rescored, err := scoreUnpairedTargets(plan, observations)
	return rescored.heldout(), err
}

func canonicalizeUnpairedTargetPlan(plan *unpairedTargetPlan) error {
	if plan == nil || plan.Version != artifact.InitialDocumentVersion || plan.View.Kind() != artifact.KindProfile ||
		plan.Objective.Kind() != artifact.KindProfile || plan.Heldout.Kind() != artifact.KindDatasetShard ||
		plan.Signature.Validate() != nil || len(plan.Signature.Outputs) != 1 ||
		plan.Signature.Outputs[0] != recipecontract.ModalityText {
		return errors.New("evaluation: invalid unpaired preference target plan")
	}
	slices.SortFunc(plan.Records, func(a, b unpairedTargetRecord) int { return cmp.Compare(a.ID, b.ID) })
	var desirable int
	for index, record := range plan.Records {
		if record.ID == "" || index > 0 && plan.Records[index-1].ID == record.ID {
			return errors.New("evaluation: invalid unpaired preference target record")
		}
		if record.Desirable {
			desirable++
		}
	}
	// The policy is KTO's with the weights the plan's own labels derive.
	if derived, err := trainingprogram.KTOPolicy(plan.Policy.Reference, plan.Policy.Scale, desirable, len(plan.Records)-desirable); err != nil || derived != plan.Policy {
		return errors.Join(errors.New("evaluation: an unpaired preference target plan's policy is KTO's over its labels"), err)
	}
	return nil
}

func canonicalizeUnpairedTargetReport(report *unpairedTargetReport) error {
	// A lone record has no other to mismatch its prompt with.
	if report == nil || report.Version != artifact.InitialDocumentVersion || report.Plan.Kind() != artifact.KindProfile ||
		len(report.Records) <= 1 || len(report.Metrics) != len(unpairedMetrics) || !checked.Finite64(report.KL) || report.KL < 0 {
		return errors.New("evaluation: invalid unpaired preference target report")
	}
	for index, record := range report.Records {
		if record.Record == "" || index > 0 && report.Records[index-1].Record >= record.Record || !checked.Finite64(record.Reward) ||
			record.Reward != record.Policy.LogProbability-record.Reference.LogProbability {
			return errors.New("evaluation: invalid unpaired preference target result")
		}
	}
	for index, metric := range unpairedMetrics {
		if report.Metrics[index].Name != metric.name || report.Metrics[index].Direction != metric.direction || !checked.Finite64(report.Metrics[index].Value) {
			return errors.New("evaluation: invalid unpaired preference target metric")
		}
	}
	return nil
}

func cloneUnpairedTargetPlan(plan unpairedTargetPlan) unpairedTargetPlan {
	plan.Signature = plan.Signature.Clone()
	plan.Records = slices.Clone(plan.Records)
	return plan
}

func cloneUnpairedTargetReport(report unpairedTargetReport) unpairedTargetReport {
	report.Records = slices.Clone(report.Records)
	report.Metrics = slices.Clone(report.Metrics)
	return report
}
