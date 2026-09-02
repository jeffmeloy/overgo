package evaluation

import (
	"context"
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

const ExactGenerationKind = "exact-generation"

type Runtime interface {
	Generator
	ContinuationScorer
}

type SuiteDescriptor struct {
	Kind    string      `json:"kind"`
	Source  string      `json:"source"`
	Plan    artifact.ID `json:"plan"`
	Dataset artifact.ID `json:"dataset"`
	Cases   uint64      `json:"cases"`
}

type SuiteResult struct {
	Report  artifact.ID
	Metrics []runrecord.Metric
}

type CompiledSuite struct {
	descriptor SuiteDescriptor
	plan       Plan
	acceptance AcceptancePolicy
	evaluator  Evaluator
	execute    func(context.Context, artifact.Repository, Runtime) (SuiteResult, error)
}

func (suite CompiledSuite) Descriptor() SuiteDescriptor { return suite.descriptor }

func (suite CompiledSuite) Plan() Plan { return suite.plan }

func CompileSuite(data []byte, authorities ExactAuthorities) (CompiledSuite, error) {
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
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateMultipleChoice(ctx, repository, runtime, compiled, plan)
				return SuiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case SequenceScoringKind:
		var source SequenceScoringSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileSequenceScoring(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := BindSequenceScoring(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err,
			sequenceScoringMetrics(0, 1),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateSequenceScoring(ctx, repository, runtime, compiled, plan)
				return SuiteResult{
					Report:  report.ID,
					Metrics: sequenceScoringMetrics(report.MeanNLLPerToken, report.Perplexity),
				}, err
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
		plan, err := BindGeneratedAnswer(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateGeneratedAnswer(ctx, repository, runtime, compiled, plan)
				return SuiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case MMLUProKind:
		var source MMLUProSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileMMLUPro(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := BindMMLUPro(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetricContract(compiled.categories),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateMMLUPro(ctx, repository, runtime, compiled, plan)
				return SuiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, report.Categories)}, err
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
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateGroupedChoice(ctx, repository, runtime, compiled, plan)
				return SuiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, report.Groups)}, err
			})
	case ProbabilityMassKind:
		var source ProbabilityMassSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileProbabilityMass(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := BindProbabilityMass(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, []runrecord.Metric{{
			Name: "positive-probability-mass", Direction: runrecord.DirectionMaximize,
		}},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateProbabilityMass(ctx, repository, runtime, compiled, plan)
				return SuiteResult{Report: report.ID, Metrics: []runrecord.Metric{{
					Name: "positive-probability-mass", Value: report.Mean, Direction: runrecord.DirectionMaximize,
				}}}, err
			})
	case StructuredGeneratedKind:
		var source StructuredGeneratedSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileStructuredGenerated(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := BindStructuredGenerated(compiled, authorities)
		groups := make([]string, len(source.Cases))
		for index, testCase := range source.Cases {
			groups[index] = testCase.Group
		}
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, accuracyMetricContract(groups),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateStructuredGenerated(ctx, repository, runtime, compiled, plan)
				return SuiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, report.Groups)}, err
			})
	case InstructionRulesKind:
		var source InstructionRulesSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := CompileInstructionRules(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := BindInstructionRules(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, []runrecord.Metric{
			{Name: "prompt-strict", Direction: runrecord.DirectionMaximize},
			{Name: "instruction-strict", Direction: runrecord.DirectionMaximize},
			{Name: "prompt-loose", Direction: runrecord.DirectionMaximize},
			{Name: "instruction-loose", Direction: runrecord.DirectionMaximize},
		},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateInstructionRules(ctx, repository, runtime, compiled, plan)
				return SuiteResult{Report: report.ID, Metrics: []runrecord.Metric{
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
		return newCompiledSuite(ExactGenerationKind, source.Source, uint64(len(source.Cases)), plan, err, []runrecord.Metric{{
			Name: exactMetricName, Direction: runrecord.DirectionMaximize,
		}},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (SuiteResult, error) {
				report, err := EvaluateExactSharded(ctx, repository, runtime, compiled, plan, nil)
				return SuiteResult{Report: report, Metrics: []runrecord.Metric{{
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
	execute func(context.Context, artifact.Repository, Runtime) (SuiteResult, error),
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

func ExecuteSuite(ctx context.Context, repository artifact.Repository, runtime Runtime, suite CompiledSuite) (SuiteResult, error) {
	if ctx == nil || repository == nil || runtime == nil || suite.execute == nil {
		return SuiteResult{}, errors.New("evaluation: incomplete suite execution")
	}
	return suite.execute(ctx, repository, runtime)
}

func accuracyMetrics(accuracy float64, groups []AccuracyGroup) []runrecord.Metric {
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
