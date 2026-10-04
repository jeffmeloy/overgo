package evaluation

import (
	"cmp"
	"errors"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
)

const (
	textTargetPlanMediaType   = "application/vnd.overgo.text-target-plan+json"
	textTargetPlanSchema      = "overgo/text-target-plan/v1"
	textTargetReportMediaType = "application/vnd.overgo.text-target-report+json"
	textTargetReportSchema    = "overgo/text-target-report/v1"
)

// textScorer names a text target metric; every text metric is maximized.
type textScorer string

const (
	// textContinuationLikelihood scores the observed continuation log-likelihood.
	textContinuationLikelihood textScorer = "continuation-likelihood"
	// textGeneratedAnswer scores 1 when the normalized output matches any normalized answer.
	textGeneratedAnswer textScorer = "generated-answer"
	// textStructuredValidity scores the verifier's pass for the output.
	textStructuredValidity textScorer = "structured-validity"
	// textOCRFields scores the fraction of fields whose verifier value matches after normalization.
	textOCRFields textScorer = "ocr-fields"
	// textVQARules scores the fraction of rules the verifier passes.
	textVQARules textScorer = "vqa-rules"
)

// textNormalization names an answer normalization step for answer and OCR-field scorers.
type textNormalization string

const (
	// textIdentityNormalization leaves text unchanged.
	textIdentityNormalization textNormalization = "identity"
	// textTrimSpaceNormalization trims surrounding space.
	textTrimSpaceNormalization textNormalization = "trim-space"
	// textLowercaseNormalization lowercases text.
	textLowercaseNormalization textNormalization = "lowercase"
)

// textScorerSpec declares one text scorer, its normalization, and the verifier profile structured, OCR, and VQA scorers require.
type textScorerSpec struct {
	Kind          textScorer          `json:"kind"`
	Normalization []textNormalization `json:"normalization,omitempty"`
	Verifier      *artifact.ID        `json:"verifier,omitempty"`
}

// namedText is a named expected text value, such as an OCR field.
type namedText struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// textTargetCase binds a held-out record to accepted answers, expected fields, and rules.
type textTargetCase struct {
	Record  string      `json:"record"`
	Answers []string    `json:"answers,omitempty"`
	Fields  []namedText `json:"fields,omitempty"`
	Rules   []string    `json:"rules,omitempty"`
}

// textTargetSuite declares scorers and cases before binding to a held-out view.
type textTargetSuite struct {
	Scorers []textScorerSpec `json:"scorers"`
	Cases   []textTargetCase `json:"cases"`
}

// textTargetPlan binds a text suite to one held-out view and its text output signature.
type textTargetPlan struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	View      artifact.ID                      `json:"view"`
	Signature recipecontract.ModalitySignature `json:"signature"`
	Scorers   []textScorerSpec                 `json:"scorers"`
	Cases     []textTargetCase                 `json:"cases"`
}

// textVerifierResult records one verifier outcome, keyed by verifier and field or rule name.
type textVerifierResult struct {
	Verifier artifact.ID `json:"verifier"`
	Name     string      `json:"name,omitzero"`
	Value    string      `json:"value,omitzero"`
	Passed   bool        `json:"passed"`
}

// textTargetObservation carries one record's raw output, optional continuation log-likelihood, and verifier results.
type textTargetObservation struct {
	Record                    string               `json:"record"`
	Raw                       string               `json:"raw"`
	ContinuationLogLikelihood *float64             `json:"continuation_log_likelihood,omitempty"`
	Verifiers                 []textVerifierResult `json:"verifiers,omitempty"`
}

// textTargetReport holds sorted observations and per-scorer mean metrics for one plan.
type textTargetReport struct {
	ID           artifact.ID             `json:"-"`
	Version      uint16                  `json:"version"`
	Plan         artifact.ID             `json:"plan"`
	Observations []textTargetObservation `json:"observations"`
	Metrics      []runrecord.Metric      `json:"metrics"`
}

var (
	textTargetPlanCodec = artifact.JSONDocumentCodec(
		"text target plan", artifact.KindProfile, textTargetPlanMediaType, textTargetPlanSchema,
		canonicalizeTextTargetPlan, func(value textTargetPlan) artifact.ID { return value.ID },
		func(value *textTargetPlan, id artifact.ID) { value.ID = id }, cloneTextTargetPlan,
	)
	textTargetReportCodec = artifact.JSONDocumentCodec(
		"text target report", artifact.KindEvaluation, textTargetReportMediaType, textTargetReportSchema,
		canonicalizeTextTargetReport, func(value textTargetReport) artifact.ID { return value.ID },
		func(value *textTargetReport, id artifact.ID) { value.ID = id }, cloneTextTargetReport,
	)
)

// compileTextTargetPlan binds a text suite to a held-out dataset view.
func compileTextTargetPlan(view SFTEvaluationView, suite textTargetSuite) (textTargetPlan, error) {
	if _, err := view.Content(); err != nil {
		return textTargetPlan{}, err
	}
	if len(suite.Cases) != len(view.Records) {
		return textTargetPlan{}, errors.New("evaluation: text target cases differ from held-out view")
	}
	records := make(map[string]struct{}, len(view.Records))
	for _, record := range view.Records {
		records[record.ID] = struct{}{}
	}
	for _, target := range suite.Cases {
		if _, found := records[target.Record]; !found {
			return textTargetPlan{}, errors.New("evaluation: text target record is outside held-out view")
		}
	}
	plan := textTargetPlan{
		Version: artifact.InitialDocumentVersion, View: view.ID, Signature: view.Signature.Clone(),
		Scorers: slices.Clone(suite.Scorers), Cases: cloneTextTargetCases(suite.Cases),
	}
	return textTargetPlanCodec.New(plan)
}

// Content encodes the plan as its canonical artifact document.
func (plan textTargetPlan) Content() (artifact.Content, error) {
	return textTargetPlanCodec.Content(plan)
}

// Batch stages the plan with lineage to its view and declared verifiers.
func (plan textTargetPlan) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{plan.View}
	for _, scorer := range plan.Scorers {
		if scorer.Verifier != nil {
			parents = append(parents, *scorer.Verifier)
		}
	}
	return textTargetPlanCodec.Batch(key, plan, artifact.DependencyLineage(plan.ID, parents...), nil)
}

// scoreTextTargets evaluates text observations with declared scorers.
func scoreTextTargets(plan textTargetPlan, observations []textTargetObservation) (textTargetReport, error) {
	if err := textTargetPlanCodec.ValidateIdentity(plan); err != nil || len(observations) != len(plan.Cases) {
		return textTargetReport{}, errors.New("evaluation: text target observations differ from plan")
	}
	observations = cloneTextObservations(observations)
	slices.SortFunc(observations, func(a, b textTargetObservation) int { return cmp.Compare(a.Record, b.Record) })
	metrics := make([]runrecord.Metric, 0, len(plan.Scorers))
	for _, scorer := range plan.Scorers {
		var sum float64
		for index, observation := range observations {
			if observation.Record != plan.Cases[index].Record {
				return textTargetReport{}, errors.New("evaluation: text target record order differs")
			}
			value, err := scoreTextObservation(scorer, plan.Cases[index], observation)
			if err != nil {
				return textTargetReport{}, err
			}
			sum += value
		}
		metrics = append(metrics, runrecord.Metric{
			Name: string(scorer.Kind), Value: sum / float64(len(observations)), Direction: runrecord.DirectionMaximize,
		})
	}
	return textTargetReportCodec.NewInitial(textTargetReport{
		Plan: plan.ID, Observations: observations, Metrics: metrics,
	})
}

// Content encodes the report as its canonical artifact document.
func (report textTargetReport) Content() (artifact.Content, error) {
	return textTargetReportCodec.Content(report)
}

// Batch stages the report with lineage to its plan.
func (report textTargetReport) Batch(key string) (artifact.Batch, error) {
	return textTargetReportCodec.Batch(key, report, artifact.DependencyLineage(report.ID, report.Plan), nil)
}

// compileTextHeldout is the generated-text held-out slice: one scorer over
// each case's accepted answers, and the text each observation generated.
func compileTextHeldout(view SFTEvaluationView, scorer textScorerSpec, cases []HeldoutCase) (HeldoutPlan, error) {
	suite := textTargetSuite{Scorers: []textScorerSpec{scorer}}
	for _, target := range cases {
		suite.Cases = append(suite.Cases, textTargetCase{Record: target.Record, Answers: slices.Clone(target.Answers)})
	}
	plan, err := compileTextTargetPlan(view, suite)
	if err != nil {
		return HeldoutPlan{}, err
	}
	return HeldoutPlan{id: plan.ID, view: plan.View, batch: plan.Batch, score: func(observations []HeldoutObservation) (HeldoutReport, error) {
		generated := make([]textTargetObservation, len(observations))
		for index, observation := range observations {
			generated[index] = textTargetObservation{Record: observation.Record, Raw: observation.Text}
		}
		report, err := scoreTextTargets(plan, generated)
		if err != nil {
			return HeldoutReport{}, err
		}
		return report.heldout(), nil
	}}, nil
}

// heldout is the report as the held-out judge reads it.
func (report textTargetReport) heldout() HeldoutReport {
	return HeldoutReport{id: report.ID, plan: report.Plan, metrics: report.Metrics, batch: report.Batch}
}

// rescore scores the observations report retains under plan.
func (plan textTargetPlan) rescore(report textTargetReport) (HeldoutReport, error) {
	rescored, err := scoreTextTargets(plan, report.Observations)
	return rescored.heldout(), err
}

func scoreTextObservation(scorer textScorerSpec, target textTargetCase, observation textTargetObservation) (float64, error) {
	switch scorer.Kind {
	case textContinuationLikelihood:
		if observation.ContinuationLogLikelihood == nil || math.IsNaN(*observation.ContinuationLogLikelihood) ||
			math.IsInf(*observation.ContinuationLogLikelihood, 0) {
			return 0, errors.New("evaluation: continuation likelihood is absent or non-finite")
		}
		return *observation.ContinuationLogLikelihood, nil
	case textGeneratedAnswer:
		actual := normalizeAnswer(observation.Raw, scorer.Normalization)
		for _, answer := range target.Answers {
			if actual == normalizeAnswer(answer, scorer.Normalization) {
				return 1, nil
			}
		}
		return 0, nil
	case textStructuredValidity:
		return verifierPass(*scorer.Verifier, "", observation.Verifiers)
	case textOCRFields:
		if len(target.Fields) == 0 {
			return 0, errors.New("evaluation: OCR field targets are absent")
		}
		var passed float64
		for _, field := range target.Fields {
			value, err := verifierValue(*scorer.Verifier, field.Name, observation.Verifiers)
			if err != nil {
				return 0, err
			}
			if normalizeAnswer(value, scorer.Normalization) == normalizeAnswer(field.Value, scorer.Normalization) {
				passed++
			}
		}
		return passed / float64(len(target.Fields)), nil
	case textVQARules:
		if len(target.Rules) == 0 {
			return 0, errors.New("evaluation: VQA rules are absent")
		}
		var passed float64
		for _, rule := range target.Rules {
			value, err := verifierPass(*scorer.Verifier, rule, observation.Verifiers)
			if err != nil {
				return 0, err
			}
			passed += value
		}
		return passed / float64(len(target.Rules)), nil
	default:
		return 0, errors.New("evaluation: unknown text scorer")
	}
}

func verifierPass(verifier artifact.ID, name string, values []textVerifierResult) (float64, error) {
	for _, value := range values {
		if value.Verifier == verifier && value.Name == name {
			if value.Passed {
				return 1, nil
			}
			return 0, nil
		}
	}
	return 0, errors.New("evaluation: declared text verifier result is absent")
}

func verifierValue(verifier artifact.ID, name string, values []textVerifierResult) (string, error) {
	for _, value := range values {
		if value.Verifier == verifier && value.Name == name {
			return value.Value, nil
		}
	}
	return "", errors.New("evaluation: declared text verifier field is absent")
}

func canonicalizeTextTargetPlan(plan *textTargetPlan) error {
	if plan == nil || plan.Version != artifact.InitialDocumentVersion || plan.View.Kind() != artifact.KindProfile ||
		plan.Signature.Validate() != nil || len(plan.Signature.Outputs) != 1 || plan.Signature.Outputs[0] != recipecontract.ModalityText ||
		len(plan.Scorers) == 0 || len(plan.Cases) == 0 {
		return errors.New("evaluation: invalid text target plan")
	}
	for _, input := range plan.Signature.Inputs {
		if input != recipecontract.ModalityText && input != recipecontract.ModalityImage &&
			input != recipecontract.ModalityAudio && input != recipecontract.ModalityVideo {
			return errors.New("evaluation: unsupported text target input modality")
		}
	}
	slices.SortFunc(plan.Scorers, func(a, b textScorerSpec) int { return cmp.Compare(a.Kind, b.Kind) })
	for index := range plan.Scorers {
		scorer := &plan.Scorers[index]
		if index > 0 && plan.Scorers[index-1].Kind == scorer.Kind || !validTextScorer(*scorer) {
			return errors.New("evaluation: invalid or duplicate text scorer")
		}
		scorer.Normalization = slices.Clone(scorer.Normalization)
	}
	slices.SortFunc(plan.Cases, func(a, b textTargetCase) int { return cmp.Compare(a.Record, b.Record) })
	for index := range plan.Cases {
		target := &plan.Cases[index]
		slices.SortFunc(target.Fields, func(a, b namedText) int { return cmp.Compare(a.Name, b.Name) })
		slices.Sort(target.Rules)
		if target.Record == "" || index > 0 && plan.Cases[index-1].Record == target.Record {
			return errors.New("evaluation: invalid text target case")
		}
		for fieldIndex, field := range target.Fields {
			if field.Name == "" || field.Value == "" || fieldIndex > 0 && target.Fields[fieldIndex-1].Name == field.Name {
				return errors.New("evaluation: invalid text target field")
			}
		}
		for ruleIndex, rule := range target.Rules {
			if rule == "" || ruleIndex > 0 && target.Rules[ruleIndex-1] == rule {
				return errors.New("evaluation: invalid text target rule")
			}
		}
	}
	for _, scorer := range plan.Scorers {
		for _, target := range plan.Cases {
			if scorer.Kind == textGeneratedAnswer && len(target.Answers) == 0 ||
				scorer.Kind == textOCRFields && len(target.Fields) == 0 ||
				scorer.Kind == textVQARules && len(target.Rules) == 0 {
				return errors.New("evaluation: text target lacks scorer facts")
			}
		}
	}
	return nil
}

func validTextScorer(scorer textScorerSpec) bool {
	valid := scorer.Kind == textContinuationLikelihood || scorer.Kind == textGeneratedAnswer ||
		scorer.Kind == textStructuredValidity || scorer.Kind == textOCRFields || scorer.Kind == textVQARules
	if !valid || scorer.Kind != textGeneratedAnswer && scorer.Kind != textOCRFields && len(scorer.Normalization) != 0 {
		return false
	}
	for _, operation := range scorer.Normalization {
		if operation != textIdentityNormalization && operation != textTrimSpaceNormalization && operation != textLowercaseNormalization {
			return false
		}
	}
	requiresVerifier := scorer.Kind == textStructuredValidity || scorer.Kind == textOCRFields || scorer.Kind == textVQARules
	return requiresVerifier == (scorer.Verifier != nil && scorer.Verifier.Kind() == artifact.KindProfile)
}

func canonicalizeTextTargetReport(report *textTargetReport) error {
	if report == nil || report.Version != artifact.InitialDocumentVersion || report.Plan.Kind() != artifact.KindProfile ||
		len(report.Observations) == 0 || len(report.Metrics) == 0 {
		return errors.New("evaluation: invalid text target report")
	}
	for index, observation := range report.Observations {
		if observation.Record == "" || index > 0 && report.Observations[index-1].Record >= observation.Record {
			return errors.New("evaluation: invalid text target observation")
		}
	}
	for index, metric := range report.Metrics {
		if metric.Name == "" || metric.Direction != runrecord.DirectionMaximize || metric.Unit != "" ||
			math.IsNaN(metric.Value) || math.IsInf(metric.Value, 0) ||
			index > 0 && report.Metrics[index-1].Name >= metric.Name {
			return errors.New("evaluation: invalid text target metric")
		}
	}
	return nil
}

func cloneTextTargetPlan(plan textTargetPlan) textTargetPlan {
	plan.Signature = plan.Signature.Clone()
	plan.Scorers = slices.Clone(plan.Scorers)
	for index := range plan.Scorers {
		plan.Scorers[index].Normalization = slices.Clone(plan.Scorers[index].Normalization)
		if plan.Scorers[index].Verifier != nil {
			value := *plan.Scorers[index].Verifier
			plan.Scorers[index].Verifier = &value
		}
	}
	plan.Cases = cloneTextTargetCases(plan.Cases)
	return plan
}

func cloneTextTargetCases(cases []textTargetCase) []textTargetCase {
	result := slices.Clone(cases)
	for index := range result {
		result[index].Answers = slices.Clone(result[index].Answers)
		result[index].Fields = slices.Clone(result[index].Fields)
		result[index].Rules = slices.Clone(result[index].Rules)
	}
	return result
}

func cloneTextObservations(values []textTargetObservation) []textTargetObservation {
	result := slices.Clone(values)
	for index := range result {
		result[index].Verifiers = slices.Clone(result[index].Verifiers)
	}
	return result
}

func cloneTextTargetReport(report textTargetReport) textTargetReport {
	report.Observations = cloneTextObservations(report.Observations)
	report.Metrics = slices.Clone(report.Metrics)
	return report
}
