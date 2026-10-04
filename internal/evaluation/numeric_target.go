package evaluation

import (
	"cmp"
	"errors"
	"math"
	"slices"

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

// numericScorer names a numeric target metric.
type numericScorer string

const (
	// numericAbsoluteError scores mean absolute residual over target values; minimized.
	numericAbsoluteError numericScorer = "absolute-error"
	// numericSquaredError scores mean squared residual over target values; minimized.
	numericSquaredError numericScorer = "squared-error"
	// numericQuantileLoss scores mean pinball loss at the spec quantile; minimized.
	numericQuantileLoss numericScorer = "quantile-loss"
	// numericLabelAccuracy scores exact label match as 1 or 0, or for a record
	// of several rows given as class ids the fraction of rows matched; maximized.
	numericLabelAccuracy numericScorer = "label-accuracy"
)

// numericScorerSpec declares one numeric scorer with its unit and, for quantile loss only, a quantile in (0, 1).
type numericScorerSpec struct {
	Kind     numericScorer `json:"kind"`
	Unit     string        `json:"unit,omitzero"`
	Quantile *float64      `json:"quantile,omitempty"`
}

// numericAssumption is a named stated assumption carried with a target case.
type numericAssumption struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// numericTargetCase binds a held-out record to expected values or label and stated assumptions.
type numericTargetCase struct {
	Record      string              `json:"record"`
	Values      []float64           `json:"values,omitempty"`
	Label       string              `json:"label,omitzero"`
	Assumptions []numericAssumption `json:"assumptions,omitempty"`
}

// numericTargetSuite declares scorers and cases before binding to a held-out view.
type numericTargetSuite struct {
	Scorers []numericScorerSpec `json:"scorers"`
	Cases   []numericTargetCase `json:"cases"`
}

// numericTargetPlan binds a numeric suite to one held-out view and its time-series or table output signature.
type numericTargetPlan struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	View      artifact.ID                      `json:"view"`
	Signature recipecontract.ModalitySignature `json:"signature"`
	Scorers   []numericScorerSpec              `json:"scorers"`
	Cases     []numericTargetCase              `json:"cases"`
}

// numericTargetObservation carries one record's predicted values or label.
type numericTargetObservation struct {
	Record string    `json:"record"`
	Values []float64 `json:"values,omitempty"`
	Label  string    `json:"label,omitzero"`
}

// numericScore is one scorer's value for one record.
type numericScore struct {
	Name  numericScorer `json:"name"`
	Value float64       `json:"value"`
}

// numericRecordResult pairs a case's expected values and label with the observed ones and per-scorer scores.
type numericRecordResult struct {
	Record        string              `json:"record"`
	Expected      []float64           `json:"expected,omitempty"`
	Actual        []float64           `json:"actual,omitempty"`
	ExpectedLabel string              `json:"expected_label,omitzero"`
	ActualLabel   string              `json:"actual_label,omitzero"`
	Assumptions   []numericAssumption `json:"assumptions,omitempty"`
	Scores        []numericScore      `json:"scores"`
}

// numericTargetReport holds per-record numeric results and per-scorer mean metrics for one plan.
type numericTargetReport struct {
	ID      artifact.ID           `json:"-"`
	Version uint16                `json:"version"`
	Plan    artifact.ID           `json:"plan"`
	Records []numericRecordResult `json:"records"`
	Metrics []runrecord.Metric    `json:"metrics"`
}

var (
	numericTargetPlanCodec = artifact.JSONDocumentCodec(
		"numeric target plan", artifact.KindProfile, numericTargetPlanMedia, numericTargetPlanSchema,
		canonicalizeNumericTargetPlan, func(value numericTargetPlan) artifact.ID { return value.ID },
		func(value *numericTargetPlan, id artifact.ID) { value.ID = id }, cloneNumericTargetPlan,
	)
	numericTargetReportCodec = artifact.JSONDocumentCodec(
		"numeric target report", artifact.KindEvaluation, numericTargetReportMedia, numericTargetReportSchema,
		canonicalizeNumericTargetReport, func(value numericTargetReport) artifact.ID { return value.ID },
		func(value *numericTargetReport, id artifact.ID) { value.ID = id }, cloneNumericTargetReport,
	)
)

// compileNumericTargetPlan binds a numeric suite to a held-out dataset view.
func compileNumericTargetPlan(view SFTEvaluationView, suite numericTargetSuite) (numericTargetPlan, error) {
	if _, err := view.Content(); err != nil {
		return numericTargetPlan{}, err
	}
	if len(suite.Cases) != len(view.Records) {
		return numericTargetPlan{}, errors.New("evaluation: numeric target cases differ from held-out view")
	}
	records := make(map[string]struct{}, len(view.Records))
	for _, record := range view.Records {
		records[record.ID] = struct{}{}
	}
	for _, target := range suite.Cases {
		if _, found := records[target.Record]; !found {
			return numericTargetPlan{}, errors.New("evaluation: numeric target record is outside held-out view")
		}
	}
	return numericTargetPlanCodec.NewInitial(numericTargetPlan{
		View: view.ID, Signature: view.Signature.Clone(),
		Scorers: slices.Clone(suite.Scorers), Cases: cloneNumericCases(suite.Cases),
	})
}

// Batch stages the plan with lineage to its view.
func (plan numericTargetPlan) Batch(key string) (artifact.Batch, error) {
	return numericTargetPlanCodec.Batch(key, plan, artifact.DependencyLineage(plan.ID, plan.View), nil)
}

// scoreNumericTargets evaluates numeric observations with declared scorers.
func scoreNumericTargets(plan numericTargetPlan, observations []numericTargetObservation) (numericTargetReport, error) {
	if err := numericTargetPlanCodec.ValidateIdentity(plan); err != nil || len(observations) != len(plan.Cases) {
		return numericTargetReport{}, errors.New("evaluation: numeric target observations differ from plan")
	}
	observations = cloneNumericObservations(observations)
	slices.SortFunc(observations, func(a, b numericTargetObservation) int { return cmp.Compare(a.Record, b.Record) })
	report := numericTargetReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.ID,
		Records: make([]numericRecordResult, len(plan.Cases)), Metrics: make([]runrecord.Metric, len(plan.Scorers)),
	}
	for index, target := range plan.Cases {
		if observations[index].Record != target.Record {
			return numericTargetReport{}, errors.New("evaluation: numeric target records differ")
		}
		result := numericRecordResult{
			Record: target.Record, Expected: slices.Clone(target.Values), Actual: slices.Clone(observations[index].Values),
			ExpectedLabel: target.Label, ActualLabel: observations[index].Label,
			Assumptions: slices.Clone(target.Assumptions), Scores: make([]numericScore, len(plan.Scorers)),
		}
		for scorerIndex, scorer := range plan.Scorers {
			value, err := scoreNumericRecord(scorer, target, observations[index])
			if err != nil {
				return numericTargetReport{}, err
			}
			result.Scores[scorerIndex] = numericScore{Name: scorer.Kind, Value: value}
			report.Metrics[scorerIndex].Value += value
		}
		report.Records[index] = result
	}
	for index, scorer := range plan.Scorers {
		direction := runrecord.DirectionMinimize
		if scorer.Kind == numericLabelAccuracy {
			direction = runrecord.DirectionMaximize
		}
		report.Metrics[index] = runrecord.Metric{
			Name: string(scorer.Kind), Value: report.Metrics[index].Value / float64(len(report.Records)),
			Unit: scorer.Unit, Direction: direction,
		}
	}
	return numericTargetReportCodec.New(report)
}

// Batch stages the report with lineage to its plan.
func (report numericTargetReport) Batch(key string) (artifact.Batch, error) {
	return numericTargetReportCodec.Batch(key, report, artifact.DependencyLineage(report.ID, report.Plan), nil)
}

// compileNumericHeldout is the numeric held-out slice: one scorer over each
// case's target values or label, and each observation's predicted ones.
func compileNumericHeldout(view SFTEvaluationView, scorer numericScorer, cases []HeldoutCase) (HeldoutPlan, error) {
	suite := numericTargetSuite{Scorers: []numericScorerSpec{{Kind: scorer}}}
	for _, target := range cases {
		suite.Cases = append(suite.Cases, numericTargetCase{Record: target.Record, Values: slices.Clone(target.Values), Label: target.Label})
	}
	plan, err := compileNumericTargetPlan(view, suite)
	if err != nil {
		return HeldoutPlan{}, err
	}
	return HeldoutPlan{id: plan.ID, view: plan.View, batch: plan.Batch, score: func(observations []HeldoutObservation) (HeldoutReport, error) {
		predicted := make([]numericTargetObservation, len(observations))
		for index, observation := range observations {
			predicted[index] = numericTargetObservation{Record: observation.Record, Values: observation.Values, Label: observation.Label}
		}
		report, err := scoreNumericTargets(plan, predicted)
		if err != nil {
			return HeldoutReport{}, err
		}
		return report.heldout(), nil
	}}, nil
}

// heldout is the report as the held-out judge reads it.
func (report numericTargetReport) heldout() HeldoutReport {
	return HeldoutReport{id: report.ID, plan: report.Plan, metrics: report.Metrics, batch: report.Batch}
}

// rescore scores the answers each of report's records retains under plan.
func (plan numericTargetPlan) rescore(report numericTargetReport) (HeldoutReport, error) {
	observations := make([]numericTargetObservation, len(report.Records))
	for index, record := range report.Records {
		observations[index] = numericTargetObservation{Record: record.Record, Values: record.Actual, Label: record.ActualLabel}
	}
	rescored, err := scoreNumericTargets(plan, observations)
	return rescored.heldout(), err
}

func scoreNumericRecord(scorer numericScorerSpec, target numericTargetCase, observation numericTargetObservation) (float64, error) {
	if scorer.Kind == numericLabelAccuracy {
		if target.Label != "" {
			if target.Label == observation.Label {
				return 1, nil
			}
			return 0, nil
		}
		// A record of several rows scores the fraction whose predicted
		// class matches.
		if len(observation.Values) != len(target.Values) {
			return 0, errors.New("evaluation: predicted class count differs from the record's rows")
		}
		matched := 0
		for index, class := range target.Values {
			if observation.Values[index] == class {
				matched++
			}
		}
		return float64(matched) / float64(len(target.Values)), nil
	}
	if len(target.Values) == 0 || len(target.Values) != len(observation.Values) {
		return 0, errors.New("evaluation: numeric value cardinality differs")
	}
	var sum float64
	for index, expected := range target.Values {
		actual := observation.Values[index]
		if !checked.Finite64(expected) || !checked.Finite64(actual) {
			return 0, errors.New("evaluation: numeric target value is non-finite")
		}
		residual := expected - actual
		switch scorer.Kind {
		case numericAbsoluteError:
			sum += math.Abs(residual)
		case numericSquaredError:
			sum += residual * residual
		default:
			q := *scorer.Quantile
			sum += max(q*residual, (q-1)*residual)
		}
	}
	return sum / float64(len(target.Values)), nil
}

func canonicalizeNumericTargetPlan(plan *numericTargetPlan) error {
	if plan == nil || plan.Version != artifact.InitialDocumentVersion || plan.View.Kind() != artifact.KindProfile ||
		plan.Signature.Validate() != nil || len(plan.Signature.Inputs) != 1 || len(plan.Signature.Outputs) != 1 ||
		len(plan.Scorers) == 0 || len(plan.Cases) == 0 {
		return errors.New("evaluation: invalid numeric target plan")
	}
	// Text scores only as label accuracy over token ids: the fraction of
	// next tokens predicted. A numeric error has no meaning over token ids.
	modality := plan.Signature.Outputs[0]
	tokenAccuracy := modality == recipecontract.ModalityText && !slices.ContainsFunc(plan.Scorers, func(scorer numericScorerSpec) bool {
		return scorer.Kind != numericLabelAccuracy
	})
	if modality != recipecontract.ModalityTimeSeries && modality != recipecontract.ModalityTable && !tokenAccuracy {
		return errors.New("evaluation: unsupported numeric target modality")
	}
	slices.SortFunc(plan.Scorers, func(a, b numericScorerSpec) int { return cmp.Compare(a.Kind, b.Kind) })
	for index, scorer := range plan.Scorers {
		if !validNumericScorer(scorer) || index > 0 && plan.Scorers[index-1].Kind == scorer.Kind {
			return errors.New("evaluation: invalid or duplicate numeric scorer")
		}
	}
	slices.SortFunc(plan.Cases, func(a, b numericTargetCase) int { return cmp.Compare(a.Record, b.Record) })
	for index := range plan.Cases {
		target := &plan.Cases[index]
		slices.SortFunc(target.Assumptions, func(a, b numericAssumption) int { return cmp.Compare(a.Name, b.Name) })
		if target.Record == "" || index > 0 && plan.Cases[index-1].Record == target.Record || !validNumericCase(*target, plan.Scorers) {
			return errors.New("evaluation: invalid numeric target case")
		}
	}
	return nil
}

func validNumericScorer(scorer numericScorerSpec) bool {
	if scorer.Kind == numericQuantileLoss {
		return scorer.Quantile != nil && checked.Finite64(*scorer.Quantile) && *scorer.Quantile > 0 && *scorer.Quantile < 1
	}
	return (scorer.Kind == numericAbsoluteError || scorer.Kind == numericSquaredError || scorer.Kind == numericLabelAccuracy) && scorer.Quantile == nil &&
		(scorer.Kind != numericLabelAccuracy || scorer.Unit == "")
}

func validNumericCase(target numericTargetCase, scorers []numericScorerSpec) bool {
	for index, assumption := range target.Assumptions {
		if assumption.Name == "" || assumption.Value == "" || index > 0 && target.Assumptions[index-1].Name == assumption.Name {
			return false
		}
	}
	for _, scorer := range scorers {
		if scorer.Kind == numericLabelAccuracy && target.Label == "" && !classIDs(target.Values) ||
			scorer.Kind != numericLabelAccuracy && len(target.Values) == 0 {
			return false
		}
	}
	return true
}

// classIDs reports whether values name one class per row: non-empty, and
// each a non-negative integer.
func classIDs(values []float64) bool {
	return len(values) != 0 && !slices.ContainsFunc(values, func(value float64) bool {
		return !checked.Finite64(value) || value < 0 || value != math.Trunc(value)
	})
}

func canonicalizeNumericTargetReport(report *numericTargetReport) error {
	if report == nil || report.Version != artifact.InitialDocumentVersion || report.Plan.Kind() != artifact.KindProfile ||
		len(report.Records) == 0 || len(report.Metrics) == 0 {
		return errors.New("evaluation: invalid numeric target report")
	}
	for index, record := range report.Records {
		if record.Record == "" || index > 0 && report.Records[index-1].Record >= record.Record || len(record.Scores) != len(report.Metrics) {
			return errors.New("evaluation: invalid numeric target result")
		}
		for _, score := range record.Scores {
			if score.Name == "" || !checked.Finite64(score.Value) {
				return errors.New("evaluation: invalid numeric record score")
			}
		}
	}
	for index, metric := range report.Metrics {
		if metric.Name == "" || !checked.Finite64(metric.Value) || index > 0 && report.Metrics[index-1].Name >= metric.Name ||
			metric.Direction != runrecord.DirectionMinimize && metric.Direction != runrecord.DirectionMaximize {
			return errors.New("evaluation: invalid numeric target metric")
		}
	}
	return nil
}

func cloneNumericCases(values []numericTargetCase) []numericTargetCase {
	result := slices.Clone(values)
	for index := range result {
		result[index].Values = slices.Clone(result[index].Values)
		result[index].Assumptions = slices.Clone(result[index].Assumptions)
	}
	return result
}

func cloneNumericTargetPlan(plan numericTargetPlan) numericTargetPlan {
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

func cloneNumericObservations(values []numericTargetObservation) []numericTargetObservation {
	result := slices.Clone(values)
	for index := range result {
		result[index].Values = slices.Clone(result[index].Values)
	}
	return result
}

func cloneNumericTargetReport(report numericTargetReport) numericTargetReport {
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
