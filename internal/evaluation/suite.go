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
	// Chance is what an answer drawn at random among each case's alternatives
	// scores, for a suite whose every case is a choice among options or
	// candidates; zero for any other. It is read where a suite is compiled and
	// is not part of the listing a client reads.
	Chance float64 `json:"-"`
}

// EvaluatedSet is what a stored plan's case profile says of the set the plan
// was run over. Measure names the measure the set was built for: the source
// its builder gave it and the cases it asks about, by their names alone. A
// set's identity is that of its whole cases, so a measure built again, over
// clips a model drew again, is another set though it asks the same cases;
// this name is what says the two are one measure. The source alone would not:
// one builder gives one source to the sets it builds at different sizes and
// lengths, which ask different cases and are different measures. Cases is how
// many the set holds, and Chance what a suite descriptor's is.
type EvaluatedSet struct {
	Measure string
	Cases   uint64
	Chance  float64
}

// EvaluatedMeasure reads the set a stored plan was run over. A plan the store
// cannot read, or a profile without a source or with a case that has no name,
// reports false.
func EvaluatedMeasure(ctx context.Context, reader artifact.Reader, plan artifact.ID) (EvaluatedSet, bool) {
	stored, err := loadEvidencePlan(ctx, reader, plan)
	if err != nil {
		return EvaluatedSet{}, false
	}
	profile, found, err := artifact.ReadContent(ctx, reader, stored.body.CaseProfile)
	if err != nil || !found {
		return EvaluatedSet{}, false
	}
	var named struct {
		Source string `json:"source"`
		Cases  []struct {
			Name string `json:"name"`
		} `json:"cases"`
	}
	if json.Unmarshal(profile.Data, &named) != nil || strings.TrimSpace(named.Source) == "" || len(named.Cases) == 0 {
		return EvaluatedSet{}, false
	}
	asked := sha256.New()
	for _, testCase := range named.Cases {
		if testCase.Name == "" {
			return EvaluatedSet{}, false
		}
		asked.Write([]byte(testCase.Name))
		asked.Write([]byte{'\n'})
	}
	return EvaluatedSet{
		Measure: named.Source + " " + hex.EncodeToString(asked.Sum(nil)), Cases: uint64(len(named.Cases)), Chance: choiceChance(profile.Data),
	}, true
}

// choiceChance is what an answer drawn at random among each case's
// alternatives scores over a suite document's cases: the mean, over the
// cases, of one over the options or candidates a case offers. A suite one of
// whose cases offers fewer than two, as a case that is answered in words
// does, or names no one of them as its answer, as a case that weighs several
// does, is not a suite of choices, and has none.
func choiceChance(suite []byte) float64 {
	var choices struct {
		Cases []struct {
			Options    []json.RawMessage `json:"options"`
			Candidates []json.RawMessage `json:"candidates"`
			Answer     json.RawMessage   `json:"answer"`
		} `json:"cases"`
	}
	if json.Unmarshal(suite, &choices) != nil || len(choices.Cases) == 0 {
		return 0
	}
	var chance float64
	for _, testCase := range choices.Cases {
		alternatives := len(testCase.Options) + len(testCase.Candidates)
		if alternatives <= 1 || len(testCase.Answer) == 0 {
			return 0
		}
		chance += 1 / float64(alternatives)
	}
	return chance / float64(len(choices.Cases))
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
	suite.descriptor.Chance = choiceChance(data)
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
	case carriedContinuationKind:
		var source CarriedContinuationSuite
		if err := strictjson.DecodeBytes(data, &source); err != nil {
			return CompiledSuite{}, err
		}
		compiled, err := compileCarriedContinuation(source)
		if err != nil {
			return CompiledSuite{}, err
		}
		plan, err := bindCarriedContinuation(compiled, authorities)
		return newCompiledSuite(envelope.Kind, source.Source, uint64(len(source.Cases)), plan, err, carriedContinuationMetrics(0, 0, 0),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
				report, err := evaluateCarriedContinuation(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: carriedContinuationMetrics(report.CarriedGain, report.CarriedLoss, report.ResetLoss)}, err
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
