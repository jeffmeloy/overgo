package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const exactGenerationKind = "exact-generation"

type Runtime interface {
	Generator
	continuationScorer
}

type SuiteDescriptor struct {
	Kind    string      `json:"kind"`
	Source  string      `json:"source"`
	Plan    artifact.ID `json:"plan"`
	Dataset artifact.ID `json:"dataset"`
	Cases   uint64      `json:"cases"`
}

// EvaluatedMeasure names the measure a stored plan was run for, from the
// plan's case profile: the source its builder gave it, and the cases it asks
// about, by their names alone. A set's identity is that of its whole cases, so
// a measure built again, over clips a model drew again, is another set though
// it asks the same cases; this name is what says the two are one measure. The
// source alone would not: one builder gives one source to the sets it builds at
// different sizes and lengths, which ask different cases and are different
// measures. With the name comes how many cases the set holds. A plan the store
// cannot read, or a profile without a source or with a case that has no name,
// reports false.
func EvaluatedMeasure(ctx context.Context, reader artifact.Reader, plan artifact.ID) (measure string, cases uint64, ok bool) {
	stored, err := loadEvidencePlan(ctx, reader, plan)
	if err != nil {
		return "", 0, false
	}
	profile, found, err := artifact.ReadContent(ctx, reader, stored.body.CaseProfile)
	if err != nil || !found {
		return "", 0, false
	}
	var named struct {
		Source string `json:"source"`
		Cases  []struct {
			Name string `json:"name"`
		} `json:"cases"`
	}
	if json.Unmarshal(profile.Data, &named) != nil || strings.TrimSpace(named.Source) == "" || len(named.Cases) == 0 {
		return "", 0, false
	}
	asked := sha256.New()
	for _, testCase := range named.Cases {
		if testCase.Name == "" {
			return "", 0, false
		}
		asked.Write([]byte(testCase.Name))
		asked.Write([]byte{'\n'})
	}
	return named.Source + " " + hex.EncodeToString(asked.Sum(nil)), uint64(len(named.Cases)), true
}

type suiteResult struct {
	Report  artifact.ID
	Metrics []runrecord.Metric
}

type CompiledSuite struct {
	descriptor SuiteDescriptor
	plan       Plan
	acceptance AcceptancePolicy
	evaluator  Evaluator
	execute    func(context.Context, artifact.Repository, Runtime) (suiteResult, error)
	source     []byte // the suite document, which a replay samples
}

func (suite CompiledSuite) Descriptor() SuiteDescriptor { return suite.descriptor }

func (suite CompiledSuite) Plan() Plan { return suite.plan }

func CompileSuite(data []byte, authorities ExactAuthorities) (CompiledSuite, error) {
	suite, err := compileSuite(data, authorities)
	if err != nil {
		return CompiledSuite{}, err
	}
	suite.source = data
	return suite, nil
}

func compileSuite(data []byte, authorities ExactAuthorities) (CompiledSuite, error) {
	var envelope struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return CompiledSuite{}, fmt.Errorf("evaluation: decode suite envelope: %w", err)
	}
	switch envelope.Kind {
	case MultipleChoiceKind:
		var source MultipleChoiceSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileMultipleChoice(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := BindMultipleChoice(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateMultipleChoice(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case sequenceScoringKind:
		var source sequenceScoringSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileSequenceScoring(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindSequenceScoring(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err,
			sequenceScoringMetrics(0, 1),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateSequenceScoring(ctx, repository, runtime, compiled, plan)
				return suiteResult{
					Report:  report.ID,
					Metrics: sequenceScoringMetrics(report.MeanNLLPerToken, report.Perplexity),
				}, err
			})
	case imageRetrievalKind:
		var source ImageRetrievalSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileImageRetrieval(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindImageRetrieval(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateImageRetrieval(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case reconstructionKind:
		var source ReconstructionSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileReconstruction(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindReconstruction(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, reconstructionMetrics(0),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateReconstruction(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: reconstructionMetrics(report.PSNR)}, err
			})
	case speechRoundTripKind:
		var source SpeechRoundTripSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileSpeechRoundTrip(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindSpeechRoundTrip(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, speechRoundTripMetrics(0),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateSpeechRoundTrip(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: speechRoundTripMetrics(report.WordErrorRate)}, err
			})
	case descriptionMatchingKind:
		var source DescriptionMatchingSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileDescriptionMatching(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindDescriptionMatching(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateDescriptionMatching(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case seriesForecastKind:
		var source SeriesForecastSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileSeriesForecast(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindSeriesForecast(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, seriesForecastMetrics(0),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateSeriesForecast(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: seriesForecastMetrics(report.MeanAbsoluteError)}, err
			})
	case functionCallKind:
		var source FunctionCallSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileFunctionCall(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindFunctionCall(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateFunctionCall(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case GeneratedAnswerKind:
		var source GeneratedAnswerSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileGeneratedAnswer(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindGeneratedAnswer(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateGeneratedAnswer(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case mmluProKind:
		var source mmluProSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileMMLUPro(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindMMLUPro(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetricContract(compiled.categories),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateMMLUPro(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, report.Categories)}, err
			})
	case GroupedChoiceKind:
		var source GroupedChoiceSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileGroupedChoice(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := BindGroupedChoice(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetricContract(compiled.groups),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := EvaluateGroupedChoice(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, report.Groups)}, err
			})
	case probabilityMassKind:
		var source probabilityMassSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileProbabilityMass(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindProbabilityMass(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, []runrecord.Metric{{
			Name: "positive-probability-mass", Direction: runrecord.DirectionMaximize,
		}},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateProbabilityMass(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: []runrecord.Metric{{
					Name: "positive-probability-mass", Value: report.Mean, Direction: runrecord.DirectionMaximize,
				}}}, err
			})
	case structuredGeneratedKind:
		var source structuredGeneratedSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileStructuredGenerated(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindStructuredGenerated(compiled, authorities)
		groups := make([]string, len(source.Cases))
		for index, testCase := range source.Cases {
			groups[index] = testCase.Group
		}
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetricContract(groups),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateStructuredGenerated(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, report.Groups)}, err
			})
	case instructionRulesKind:
		var source instructionRulesSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileInstructionRules(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindInstructionRules(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, []runrecord.Metric{
			{Name: "prompt-strict", Direction: runrecord.DirectionMaximize},
			{Name: "instruction-strict", Direction: runrecord.DirectionMaximize},
			{Name: "prompt-loose", Direction: runrecord.DirectionMaximize},
			{Name: "instruction-loose", Direction: runrecord.DirectionMaximize},
		},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateInstructionRules(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: []runrecord.Metric{
					{Name: "prompt-strict", Value: report.PromptStrict, Direction: runrecord.DirectionMaximize},
					{Name: "instruction-strict", Value: report.InstructionStrict, Direction: runrecord.DirectionMaximize},
					{Name: "prompt-loose", Value: report.PromptLoose, Direction: runrecord.DirectionMaximize},
					{Name: "instruction-loose", Value: report.InstructionLoose, Direction: runrecord.DirectionMaximize},
				}}, err
			})
	case "":
		var source ExactSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileExact(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := BindExact(compiled, authorities)
		return newCompiledSuite(exactGenerationKind, source.Source, uint64(len(source.Cases)), plan, err, []runrecord.Metric{{
			Name: exactMetricName, Direction: runrecord.DirectionMaximize,
		}},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := EvaluateExactSharded(ctx, repository, runtime, compiled, plan, nil)
				return suiteResult{Report: report, Metrics: []runrecord.Metric{{
					Name: exactMetricName, Value: 1, Direction: runrecord.DirectionMaximize,
				}}}, err
			})
	default:
		return CompiledSuite{}, fmt.Errorf("evaluation: unknown suite kind %q", envelope.Kind)
	}
}

func newCompiledSuite(
	kind, source string,
	cases uint64,
	plan Plan,
	bindErr error,
	metrics []runrecord.Metric,
	execute func(context.Context, artifact.Repository, Runtime) (suiteResult, error),
) (CompiledSuite, error) {
	if bindErr != nil {
		return CompiledSuite{}, bindErr
	}
	acceptance, err := newAcceptancePolicy(metrics)
	if err != nil {
		return CompiledSuite{}, err
	}
	if execute == nil || kind == "" || source == "" || cases == 0 || !plan.Identity().Valid() {
		return CompiledSuite{}, errors.New("evaluation: incomplete compiled suite")
	}
	evaluator, err := NewEvaluator(plan.Identity(), acceptance)
	if err != nil {
		return CompiledSuite{}, err
	}
	return CompiledSuite{
		descriptor: SuiteDescriptor{Kind: kind, Source: source, Plan: plan.Identity(), Dataset: plan.Dataset(), Cases: cases},
		plan:       plan, acceptance: acceptance, evaluator: evaluator, execute: execute,
	}, nil
}

func executeSuite(ctx context.Context, repository artifact.Repository, runtime Runtime, suite CompiledSuite) (suiteResult, error) {
	if ctx == nil || repository == nil || runtime == nil || suite.execute == nil {
		return suiteResult{}, errors.New("evaluation: incomplete suite execution")
	}
	return suite.execute(ctx, repository, runtime)
}

func accuracyMetrics(accuracy float64, groups []accuracyGroup) []runrecord.Metric {
	metrics := make([]runrecord.Metric, 1, len(groups)+1)
	metrics[0] = runrecord.Metric{Name: "accuracy", Value: accuracy, Direction: runrecord.DirectionMaximize}
	for _, group := range groups {
		metrics = append(metrics, runrecord.Metric{
			Name: groupAccuracyMetricName(group.Name), Value: group.Accuracy, Direction: runrecord.DirectionMaximize,
		})
	}
	return metrics
}

// groupAccuracyMetricName names one group's accuracy metric within the
// run-record label alphabet (lower-case letters, digits, '.', '-', '_'):
// "accuracy." followed by the group name lowered, with every other byte
// written as '-'. A slash or capital in a dataset subset name must not
// make the whole evaluation unrecordable.
func groupAccuracyMetricName(group string) string {
	name := []byte("accuracy." + strings.ToLower(group))
	for index := len("accuracy."); index < len(name); index++ {
		if !textcheck.LowerIdentifierByte(name[index]) {
			name[index] = '-'
		}
	}
	return string(name)
}

func accuracyMetricContract(groups []string) []runrecord.Metric {
	names := slices.Clone(groups)
	slices.Sort(names)
	unique := names[:0]
	for _, name := range names {
		if name != "" && (len(unique) == 0 || unique[len(unique)-1] != name) {
			unique = append(unique, name)
		}
	}
	metrics := make([]runrecord.Metric, 1, len(unique)+1)
	metrics[0] = runrecord.Metric{Name: "accuracy", Direction: runrecord.DirectionMaximize}
	for _, name := range unique {
		metrics = append(metrics, runrecord.Metric{Name: groupAccuracyMetricName(name), Direction: runrecord.DirectionMaximize})
	}
	return metrics
}
