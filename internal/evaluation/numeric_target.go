package evaluation

import (
	"errors"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
)

const (
	numericTargetPlanMedia    = "application/vnd.overgo.numeric-target-plan+json"
	numericTargetPlanSchema   = "overgo/numeric-target-plan/v1"
	numericTargetReportMedia  = "application/vnd.overgo.numeric-target-report+json"
	numericTargetReportSchema = "overgo/numeric-target-report/v1"
)

type NumericScorer string

const (
	NumericAbsoluteError NumericScorer = "absolute-error"
	NumericQuantileLoss  NumericScorer = "quantile-loss"
	NumericLabelAccuracy NumericScorer = "label-accuracy"
)

type NumericScorerSpec struct {
	Kind     NumericScorer `json:"kind"`
	Unit     string        `json:"unit,omitzero"`
	Quantile *float64      `json:"quantile,omitempty"`
}

type NumericAssumption struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type NumericTargetCase struct {
	Record      string              `json:"record"`
	Values      []float64           `json:"values,omitempty"`
	Label       string              `json:"label,omitzero"`
	Assumptions []NumericAssumption `json:"assumptions,omitempty"`
}

type NumericTargetSuite struct {
	Scorers []NumericScorerSpec `json:"scorers"`
	Cases   []NumericTargetCase `json:"cases"`
}

type NumericTargetPlan struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	View      artifact.ID                      `json:"view"`
	Signature recipecontract.ModalitySignature `json:"signature"`
	Scorers   []NumericScorerSpec              `json:"scorers"`
	Cases     []NumericTargetCase              `json:"cases"`
}

type NumericTargetObservation struct {
	Record string    `json:"record"`
	Values []float64 `json:"values,omitempty"`
	Label  string    `json:"label,omitzero"`
}

type NumericScore struct {
	Name  NumericScorer `json:"name"`
	Value float64       `json:"value"`
}

type NumericRecordResult struct {
	Record        string              `json:"record"`
	Expected      []float64           `json:"expected,omitempty"`
	Actual        []float64           `json:"actual,omitempty"`
	ExpectedLabel string              `json:"expected_label,omitzero"`
	ActualLabel   string              `json:"actual_label,omitzero"`
	Assumptions   []NumericAssumption `json:"assumptions,omitempty"`
	Scores        []NumericScore      `json:"scores"`
}

type NumericTargetReport struct {
	ID      artifact.ID           `json:"-"`
	Version uint16                `json:"version"`
	Plan    artifact.ID           `json:"plan"`
	Records []NumericRecordResult `json:"records"`
	Metrics []runrecord.Metric    `json:"metrics"`
}

var (
	numericTargetPlanCodec = artifact.JSONDocumentCodec(
		"numeric target plan", artifact.KindProfile, numericTargetPlanMedia, numericTargetPlanSchema,
		canonicalizeNumericTargetPlan, func(value NumericTargetPlan) artifact.ID { return value.ID },
		func(value *NumericTargetPlan, id artifact.ID) { value.ID = id }, cloneNumericTargetPlan,
	)
	numericTargetReportCodec = artifact.JSONDocumentCodec(
		"numeric target report", artifact.KindEvaluation, numericTargetReportMedia, numericTargetReportSchema,
		canonicalizeNumericTargetReport, func(value NumericTargetReport) artifact.ID { return value.ID },
		func(value *NumericTargetReport, id artifact.ID) { value.ID = id }, cloneNumericTargetReport,
	)
)

// CompileNumericTargetPlan binds a numeric suite to a held-out dataset view.
func CompileNumericTargetPlan(view SFTEvaluationView, suite NumericTargetSuite) (NumericTargetPlan, error) {
	if _, err := view.Content(); err != nil {
		return NumericTargetPlan{}, err
	}
	if len(suite.Cases) != len(view.Records) {
		return NumericTargetPlan{}, errors.New("evaluation: numeric target cases differ from held-out view")
	}
	records := make(map[string]struct{}, len(view.Records))
	for _, record := range view.Records {
		records[record.ID] = struct{}{}
	}
	for _, target := range suite.Cases {
		if _, found := records[target.Record]; !found {
			return NumericTargetPlan{}, errors.New("evaluation: numeric target record is outside held-out view")
		}
	}
	return numericTargetPlanCodec.New(NumericTargetPlan{
		Version: artifact.InitialDocumentVersion, View: view.ID, Signature: view.Signature.Clone(),
		Scorers: slices.Clone(suite.Scorers), Cases: cloneNumericCases(suite.Cases),
	})
}

func (plan NumericTargetPlan) Batch(key string) (artifact.Batch, error) {
	return numericTargetPlanCodec.Batch(key, plan, artifact.DependencyLineage(plan.ID, plan.View), nil)
}

// ScoreNumericTargets evaluates numeric observations with declared scorers.
func ScoreNumericTargets(plan NumericTargetPlan, observations []NumericTargetObservation) (NumericTargetReport, error) {
	if err := numericTargetPlanCodec.ValidateIdentity(plan); err != nil || len(observations) != len(plan.Cases) {
		return NumericTargetReport{}, errors.New("evaluation: numeric target observations differ from plan")
	}
	observations = cloneNumericObservations(observations)
	sort.Slice(observations, func(i, j int) bool { return observations[i].Record < observations[j].Record })
	report := NumericTargetReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.ID,
		Records: make([]NumericRecordResult, len(plan.Cases)), Metrics: make([]runrecord.Metric, len(plan.Scorers)),
	}
	for index, target := range plan.Cases {
		if observations[index].Record != target.Record {
			return NumericTargetReport{}, errors.New("evaluation: numeric target records differ")
		}
		result := NumericRecordResult{
			Record: target.Record, Expected: slices.Clone(target.Values), Actual: slices.Clone(observations[index].Values),
			ExpectedLabel: target.Label, ActualLabel: observations[index].Label,
			Assumptions: slices.Clone(target.Assumptions), Scores: make([]NumericScore, len(plan.Scorers)),
		}
		for scorerIndex, scorer := range plan.Scorers {
			value, err := scoreNumericRecord(scorer, target, observations[index])
			if err != nil {
				return NumericTargetReport{}, err
			}
			result.Scores[scorerIndex] = NumericScore{Name: scorer.Kind, Value: value}
			report.Metrics[scorerIndex].Value += value
		}
		report.Records[index] = result
	}
	for index, scorer := range plan.Scorers {
		direction := runrecord.DirectionMinimize
		if scorer.Kind == NumericLabelAccuracy {
			direction = runrecord.DirectionMaximize
		}
		report.Metrics[index] = runrecord.Metric{
			Name: string(scorer.Kind), Value: report.Metrics[index].Value / float64(len(report.Records)),
			Unit: scorer.Unit, Direction: direction,
		}
	}
	return numericTargetReportCodec.New(report)
}

func (report NumericTargetReport) Batch(key string) (artifact.Batch, error) {
	return numericTargetReportCodec.Batch(key, report, artifact.DependencyLineage(report.ID, report.Plan), nil)
}

func scoreNumericRecord(scorer NumericScorerSpec, target NumericTargetCase, observation NumericTargetObservation) (float64, error) {
	if scorer.Kind == NumericLabelAccuracy {
		if target.Label == observation.Label {
			return 1, nil
		}
		return 0, nil
	}
	if len(target.Values) == 0 || len(target.Values) != len(observation.Values) {
		return 0, errors.New("evaluation: numeric value cardinality differs")
	}
	var sum float64
	for index, expected := range target.Values {
		actual := observation.Values[index]
		if !finite(expected) || !finite(actual) {
			return 0, errors.New("evaluation: numeric target value is non-finite")
		}
		residual := expected - actual
		if scorer.Kind == NumericAbsoluteError {
			sum += math.Abs(residual)
		} else {
			q := *scorer.Quantile
			sum += max(q*residual, (q-1)*residual)
		}
	}
	return sum / float64(len(target.Values)), nil
}

func canonicalizeNumericTargetPlan(plan *NumericTargetPlan) error {
	if plan == nil || plan.Version != artifact.InitialDocumentVersion || plan.View.Kind() != artifact.KindProfile ||
		plan.Signature.Validate() != nil || len(plan.Signature.Inputs) != 1 || len(plan.Signature.Outputs) != 1 ||
		len(plan.Scorers) == 0 || len(plan.Cases) == 0 {
		return errors.New("evaluation: invalid numeric target plan")
	}
	modality := plan.Signature.Outputs[0]
	if modality != recipecontract.ModalityTimeSeries && modality != recipecontract.ModalityTable {
		return errors.New("evaluation: unsupported numeric target modality")
	}
	sort.Slice(plan.Scorers, func(i, j int) bool { return plan.Scorers[i].Kind < plan.Scorers[j].Kind })
	for index, scorer := range plan.Scorers {
		if !validNumericScorer(scorer) || index > 0 && plan.Scorers[index-1].Kind == scorer.Kind {
			return errors.New("evaluation: invalid or duplicate numeric scorer")
		}
	}
	sort.Slice(plan.Cases, func(i, j int) bool { return plan.Cases[i].Record < plan.Cases[j].Record })
	for index := range plan.Cases {
		target := &plan.Cases[index]
		sort.Slice(target.Assumptions, func(i, j int) bool { return target.Assumptions[i].Name < target.Assumptions[j].Name })
		if target.Record == "" || index > 0 && plan.Cases[index-1].Record == target.Record || !validNumericCase(*target, plan.Scorers) {
			return errors.New("evaluation: invalid numeric target case")
		}
	}
	return nil
}

func validNumericScorer(scorer NumericScorerSpec) bool {
	if scorer.Kind == NumericQuantileLoss {
		return scorer.Quantile != nil && finite(*scorer.Quantile) && *scorer.Quantile > 0 && *scorer.Quantile < 1
	}
	return (scorer.Kind == NumericAbsoluteError || scorer.Kind == NumericLabelAccuracy) && scorer.Quantile == nil &&
		(scorer.Kind != NumericLabelAccuracy || scorer.Unit == "")
}

func validNumericCase(target NumericTargetCase, scorers []NumericScorerSpec) bool {
	for index, assumption := range target.Assumptions {
		if assumption.Name == "" || assumption.Value == "" || index > 0 && target.Assumptions[index-1].Name == assumption.Name {
			return false
		}
	}
	for _, scorer := range scorers {
		if scorer.Kind == NumericLabelAccuracy && target.Label == "" || scorer.Kind != NumericLabelAccuracy && len(target.Values) == 0 {
			return false
		}
	}
	return true
}

func canonicalizeNumericTargetReport(report *NumericTargetReport) error {
	if report == nil || report.Version != artifact.InitialDocumentVersion || report.Plan.Kind() != artifact.KindProfile ||
		len(report.Records) == 0 || len(report.Metrics) == 0 {
		return errors.New("evaluation: invalid numeric target report")
	}
	for index, record := range report.Records {
		if record.Record == "" || index > 0 && report.Records[index-1].Record >= record.Record || len(record.Scores) != len(report.Metrics) {
			return errors.New("evaluation: invalid numeric target result")
		}
		for _, score := range record.Scores {
			if score.Name == "" || !finite(score.Value) {
				return errors.New("evaluation: invalid numeric record score")
			}
		}
	}
	for index, metric := range report.Metrics {
		if metric.Name == "" || !finite(metric.Value) || index > 0 && report.Metrics[index-1].Name >= metric.Name ||
			metric.Direction != runrecord.DirectionMinimize && metric.Direction != runrecord.DirectionMaximize {
			return errors.New("evaluation: invalid numeric target metric")
		}
	}
	return nil
}

func finite(value float64) bool { return checked.Finite64(value) }

func cloneNumericCases(values []NumericTargetCase) []NumericTargetCase {
	result := slices.Clone(values)
	for index := range result {
		result[index].Values = slices.Clone(result[index].Values)
		result[index].Assumptions = slices.Clone(result[index].Assumptions)
	}
	return result
}

func cloneNumericTargetPlan(plan NumericTargetPlan) NumericTargetPlan {
	plan.Signature = plan.Signature.Clone()
	plan.Scorers = slices.Clone(plan.Scorers)
	for index := range plan.Scorers {
		if plan.Scorers[index].Quantile != nil {
			value := *plan.Scorers[index].Quantile
			plan.Scorers[index].Quantile = &value
		}
	}
	plan.Cases = cloneNumericCases(plan.Cases)
	return plan
}

func cloneNumericObservations(values []NumericTargetObservation) []NumericTargetObservation {
	result := slices.Clone(values)
	for index := range result {
		result[index].Values = slices.Clone(result[index].Values)
	}
	return result
}

func cloneNumericTargetReport(report NumericTargetReport) NumericTargetReport {
	report.Records = slices.Clone(report.Records)
	for index := range report.Records {
		record := &report.Records[index]
		record.Expected = slices.Clone(record.Expected)
		record.Actual = slices.Clone(record.Actual)
		record.Assumptions = slices.Clone(record.Assumptions)
		record.Scores = slices.Clone(record.Scores)
	}
	report.Metrics = slices.Clone(report.Metrics)
	return report
}
