package evaluation

import (
	"errors"
	"math"
	"slices"
	"sort"
	"strings"

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

type TextScorer string

const (
	TextContinuationLikelihood TextScorer = "continuation-likelihood"
	TextGeneratedAnswer        TextScorer = "generated-answer"
	TextStructuredValidity     TextScorer = "structured-validity"
	TextOCRFields              TextScorer = "ocr-fields"
	TextVQARules               TextScorer = "vqa-rules"
)

type TextNormalization string

const (
	TextIdentityNormalization  TextNormalization = "identity"
	TextTrimSpaceNormalization TextNormalization = "trim-space"
	TextLowercaseNormalization TextNormalization = "lowercase"
)

type TextScorerSpec struct {
	Kind          TextScorer          `json:"kind"`
	Normalization []TextNormalization `json:"normalization,omitempty"`
	Verifier      *artifact.ID        `json:"verifier,omitempty"`
}

type NamedText struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type TextTargetCase struct {
	Record  string      `json:"record"`
	Answers []string    `json:"answers,omitempty"`
	Fields  []NamedText `json:"fields,omitempty"`
	Rules   []string    `json:"rules,omitempty"`
}

type TextTargetSuite struct {
	Scorers []TextScorerSpec `json:"scorers"`
	Cases   []TextTargetCase `json:"cases"`
}

type TextTargetPlan struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	View      artifact.ID                      `json:"view"`
	Signature recipecontract.ModalitySignature `json:"signature"`
	Scorers   []TextScorerSpec                 `json:"scorers"`
	Cases     []TextTargetCase                 `json:"cases"`
}

type TextVerifierResult struct {
	Verifier artifact.ID `json:"verifier"`
	Name     string      `json:"name,omitempty"`
	Value    string      `json:"value,omitempty"`
	Passed   bool        `json:"passed"`
}

type TextTargetObservation struct {
	Record                    string               `json:"record"`
	Raw                       string               `json:"raw"`
	ContinuationLogLikelihood *float64             `json:"continuation_log_likelihood,omitempty"`
	Verifiers                 []TextVerifierResult `json:"verifiers,omitempty"`
}

type TextTargetReport struct {
	ID           artifact.ID             `json:"-"`
	Version      uint16                  `json:"version"`
	Plan         artifact.ID             `json:"plan"`
	Observations []TextTargetObservation `json:"observations"`
	Metrics      []runrecord.Metric      `json:"metrics"`
}

var (
	textTargetPlanCodec = artifact.JSONDocumentCodec(
		"text target plan", artifact.KindProfile, textTargetPlanMediaType, textTargetPlanSchema,
		canonicalizeTextTargetPlan, func(value TextTargetPlan) artifact.ID { return value.ID },
		func(value *TextTargetPlan, id artifact.ID) { value.ID = id }, cloneTextTargetPlan,
	)
	textTargetReportCodec = artifact.JSONDocumentCodec(
		"text target report", artifact.KindEvaluation, textTargetReportMediaType, textTargetReportSchema,
		canonicalizeTextTargetReport, func(value TextTargetReport) artifact.ID { return value.ID },
		func(value *TextTargetReport, id artifact.ID) { value.ID = id }, cloneTextTargetReport,
	)
)

func CompileTextTargetPlan(view SFTEvaluationView, suite TextTargetSuite) (TextTargetPlan, error) {
	if _, err := view.Content(); err != nil {
		return TextTargetPlan{}, err
	}
	if len(suite.Cases) != len(view.Records) {
		return TextTargetPlan{}, errors.New("evaluation: text target cases differ from held-out view")
	}
	records := make(map[string]struct{}, len(view.Records))
	for _, record := range view.Records {
		records[record.ID] = struct{}{}
	}
	for _, target := range suite.Cases {
		if _, found := records[target.Record]; !found {
			return TextTargetPlan{}, errors.New("evaluation: text target record is outside held-out view")
		}
	}
	plan := TextTargetPlan{
		Version: artifact.InitialDocumentVersion, View: view.ID, Signature: view.Signature.Clone(),
		Scorers: slices.Clone(suite.Scorers), Cases: cloneTextTargetCases(suite.Cases),
	}
	return textTargetPlanCodec.New(plan)
}

func (plan TextTargetPlan) Content() (artifact.Content, error) {
	return textTargetPlanCodec.Content(plan)
}

func (plan TextTargetPlan) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{plan.View}
	for _, scorer := range plan.Scorers {
		if scorer.Verifier != nil {
			parents = append(parents, *scorer.Verifier)
		}
	}
	return textTargetPlanCodec.Batch(key, plan, artifact.DependencyLineage(plan.ID, parents...), nil)
}

func ScoreTextTargets(plan TextTargetPlan, observations []TextTargetObservation) (TextTargetReport, error) {
	if err := textTargetPlanCodec.ValidateIdentity(plan); err != nil || len(observations) != len(plan.Cases) {
		return TextTargetReport{}, errors.New("evaluation: text target observations differ from plan")
	}
	observations = cloneTextObservations(observations)
	sort.Slice(observations, func(i, j int) bool { return observations[i].Record < observations[j].Record })
	metrics := make([]runrecord.Metric, 0, len(plan.Scorers))
	for _, scorer := range plan.Scorers {
		var sum float64
		for index, observation := range observations {
			if observation.Record != plan.Cases[index].Record {
				return TextTargetReport{}, errors.New("evaluation: text target record order differs")
			}
			value, err := scoreTextObservation(scorer, plan.Cases[index], observation)
			if err != nil {
				return TextTargetReport{}, err
			}
			sum += value
		}
		metrics = append(metrics, runrecord.Metric{
			Name: string(scorer.Kind), Value: sum / float64(len(observations)), Direction: runrecord.DirectionMaximize,
		})
	}
	return textTargetReportCodec.New(TextTargetReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.ID, Observations: observations, Metrics: metrics,
	})
}

func (report TextTargetReport) Content() (artifact.Content, error) {
	return textTargetReportCodec.Content(report)
}

func (report TextTargetReport) Batch(key string) (artifact.Batch, error) {
	return textTargetReportCodec.Batch(key, report, artifact.DependencyLineage(report.ID, report.Plan), nil)
}

func scoreTextObservation(scorer TextScorerSpec, target TextTargetCase, observation TextTargetObservation) (float64, error) {
	switch scorer.Kind {
	case TextContinuationLikelihood:
		if observation.ContinuationLogLikelihood == nil || math.IsNaN(*observation.ContinuationLogLikelihood) ||
			math.IsInf(*observation.ContinuationLogLikelihood, 0) {
			return 0, errors.New("evaluation: continuation likelihood is absent or non-finite")
		}
		return *observation.ContinuationLogLikelihood, nil
	case TextGeneratedAnswer:
		actual := normalizeTextTarget(observation.Raw, scorer.Normalization)
		for _, answer := range target.Answers {
			if actual == normalizeTextTarget(answer, scorer.Normalization) {
				return 1, nil
			}
		}
		return 0, nil
	case TextStructuredValidity:
		return verifierPass(*scorer.Verifier, "", observation.Verifiers)
	case TextOCRFields:
		if len(target.Fields) == 0 {
			return 0, errors.New("evaluation: OCR field targets are absent")
		}
		var passed float64
		for _, field := range target.Fields {
			value, err := verifierValue(*scorer.Verifier, field.Name, observation.Verifiers)
			if err != nil {
				return 0, err
			}
			if normalizeTextTarget(value, scorer.Normalization) == normalizeTextTarget(field.Value, scorer.Normalization) {
				passed++
			}
		}
		return passed / float64(len(target.Fields)), nil
	case TextVQARules:
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

func verifierPass(verifier artifact.ID, name string, values []TextVerifierResult) (float64, error) {
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

func verifierValue(verifier artifact.ID, name string, values []TextVerifierResult) (string, error) {
	for _, value := range values {
		if value.Verifier == verifier && value.Name == name {
			return value.Value, nil
		}
	}
	return "", errors.New("evaluation: declared text verifier field is absent")
}

func normalizeTextTarget(value string, operations []TextNormalization) string {
	for _, operation := range operations {
		switch operation {
		case TextTrimSpaceNormalization:
			value = strings.TrimSpace(value)
		case TextLowercaseNormalization:
			value = strings.ToLower(value)
		}
	}
	return value
}

func canonicalizeTextTargetPlan(plan *TextTargetPlan) error {
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
	sort.Slice(plan.Scorers, func(i, j int) bool { return plan.Scorers[i].Kind < plan.Scorers[j].Kind })
	for index := range plan.Scorers {
		scorer := &plan.Scorers[index]
		if index > 0 && plan.Scorers[index-1].Kind == scorer.Kind || !validTextScorer(*scorer) {
			return errors.New("evaluation: invalid or duplicate text scorer")
		}
		scorer.Normalization = slices.Clone(scorer.Normalization)
	}
	sort.Slice(plan.Cases, func(i, j int) bool { return plan.Cases[i].Record < plan.Cases[j].Record })
	for index := range plan.Cases {
		target := &plan.Cases[index]
		sort.Slice(target.Fields, func(i, j int) bool { return target.Fields[i].Name < target.Fields[j].Name })
		sort.Strings(target.Rules)
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
			if scorer.Kind == TextGeneratedAnswer && len(target.Answers) == 0 ||
				scorer.Kind == TextOCRFields && len(target.Fields) == 0 ||
				scorer.Kind == TextVQARules && len(target.Rules) == 0 {
				return errors.New("evaluation: text target lacks scorer facts")
			}
		}
	}
	return nil
}

func validTextScorer(scorer TextScorerSpec) bool {
	valid := scorer.Kind == TextContinuationLikelihood || scorer.Kind == TextGeneratedAnswer ||
		scorer.Kind == TextStructuredValidity || scorer.Kind == TextOCRFields || scorer.Kind == TextVQARules
	if !valid || scorer.Kind != TextGeneratedAnswer && scorer.Kind != TextOCRFields && len(scorer.Normalization) != 0 {
		return false
	}
	for _, operation := range scorer.Normalization {
		if operation != TextIdentityNormalization && operation != TextTrimSpaceNormalization && operation != TextLowercaseNormalization {
			return false
		}
	}
	requiresVerifier := scorer.Kind == TextStructuredValidity || scorer.Kind == TextOCRFields || scorer.Kind == TextVQARules
	return requiresVerifier == (scorer.Verifier != nil && scorer.Verifier.Kind() == artifact.KindProfile)
}

func canonicalizeTextTargetReport(report *TextTargetReport) error {
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

func cloneTextTargetPlan(plan TextTargetPlan) TextTargetPlan {
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

func cloneTextTargetCases(cases []TextTargetCase) []TextTargetCase {
	result := slices.Clone(cases)
	for index := range result {
		result[index].Answers = slices.Clone(result[index].Answers)
		result[index].Fields = slices.Clone(result[index].Fields)
		result[index].Rules = slices.Clone(result[index].Rules)
	}
	return result
}

func cloneTextObservations(values []TextTargetObservation) []TextTargetObservation {
	result := slices.Clone(values)
	for index := range result {
		result[index].Verifiers = slices.Clone(result[index].Verifiers)
	}
	return result
}

func cloneTextTargetReport(report TextTargetReport) TextTargetReport {
	report.Observations = cloneTextObservations(report.Observations)
	report.Metrics = slices.Clone(report.Metrics)
	return report
}
