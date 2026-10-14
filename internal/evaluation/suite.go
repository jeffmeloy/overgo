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
}

func (suite CompiledSuite) Descriptor() SuiteDescriptor { return suite.descriptor }

func (suite CompiledSuite) Plan() Plan { return suite.plan }

func CompileSuite(data []byte, authorities ExactAuthorities) (CompiledSuite, error) {
	suite, err := compileSuite(data, authorities)
	if err != nil {
		return CompiledSuite{}, err
	}
	suite.descriptor.Chance = choiceChance(data)
	return suite, nil
}

// suiteEnvelope is what a suite document of every kind holds.
//   - an exact suite holds no kind
//   - every kind's suite tags its source `source` and its cases `cases`, so the envelope reads both before a kind decodes them
type suiteEnvelope struct {
	Kind   string            `json:"kind"`
	Source string            `json:"source"`
	Cases  []json.RawMessage `json:"cases"`
}

func compileSuite(data []byte, authorities ExactAuthorities) (CompiledSuite, error) {
	var envelope suiteEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return CompiledSuite{}, fmt.Errorf("evaluation: decode suite envelope: %w", err)
	}
	switch envelope.Kind {
	case MultipleChoiceKind:
		return compileKind(data, envelope, authorities, CompileMultipleChoice, BindMultipleChoice, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled multipleChoicePlan, plan Plan) (suiteResult, error) {
				report, err := evaluateMultipleChoice(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case sequenceScoringKind:
		return compileKind(data, envelope, authorities, compileSequenceScoring, bindSequenceScoring, sequenceScoringMetrics(0, 1),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled sequenceScoringPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateSequenceScoring(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: sequenceScoringMetrics(report.MeanNLLPerToken, report.Perplexity)}, err
			})
	case imageRetrievalKind:
		return compileKind(data, envelope, authorities, compileImageRetrieval, bindImageRetrieval, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled imageRetrievalPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateImageRetrieval(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case reconstructionKind:
		return compileKind(data, envelope, authorities, compileReconstruction, bindReconstruction, reconstructionMetrics(0),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled reconstructionPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateReconstruction(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: reconstructionMetrics(report.LevelError)}, err
			})
	case speechRoundTripKind:
		return compileKind(data, envelope, authorities, compileSpeechRoundTrip, bindSpeechRoundTrip, speechRoundTripMetrics(0),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled speechRoundTripPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateSpeechRoundTrip(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: speechRoundTripMetrics(report.WordErrorRate)}, err
			})
	case descriptionMatchingKind:
		return compileKind(data, envelope, authorities, compileDescriptionMatching, bindDescriptionMatching, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled descriptionMatchingPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateDescriptionMatching(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case seriesForecastKind:
		return compileKind(data, envelope, authorities, compileSeriesForecast, bindSeriesForecast, seriesForecastMetrics(0),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled seriesForecastPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateSeriesForecast(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: seriesForecastMetrics(report.MeanAbsoluteError)}, err
			})
	case carriedContinuationKind:
		return compileKind(data, envelope, authorities, compileCarriedContinuation, bindCarriedContinuation, carriedContinuationMetrics(0, 0, 0),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled carriedContinuationPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateCarriedContinuation(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: carriedContinuationMetrics(report.CarriedGain, report.CarriedLoss, report.ResetLoss)}, err
			})
	case functionCallKind:
		return compileKind(data, envelope, authorities, compileFunctionCall, bindFunctionCall, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled functionCallPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateFunctionCall(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
			})
	case GeneratedAnswerKind:
		return compileKind(data, envelope, authorities, CompileGeneratedAnswer, bindGeneratedAnswer, accuracyMetrics(0, nil),
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled generatedAnswerPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateGeneratedAnswer(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: accuracyMetrics(report.Accuracy, nil)}, err
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
		return compileKind(data, envelope, authorities, compileProbabilityMass, bindProbabilityMass, []runrecord.Metric{{
			Name: "positive-probability-mass", Direction: runrecord.DirectionMaximize,
		}},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled probabilityMassPlan, plan Plan) (suiteResult, error) {
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
		return compileKind(data, envelope, authorities, compileInstructionRules, bindInstructionRules, []runrecord.Metric{
			{Name: "prompt-strict", Direction: runrecord.DirectionMaximize},
			{Name: "instruction-strict", Direction: runrecord.DirectionMaximize},
			{Name: "prompt-loose", Direction: runrecord.DirectionMaximize},
			{Name: "instruction-loose", Direction: runrecord.DirectionMaximize},
		},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled instructionRulesPlan, plan Plan) (suiteResult, error) {
				report, err := evaluateInstructionRules(ctx, repository, runtime, compiled, plan)
				return suiteResult{Report: report.ID, Metrics: []runrecord.Metric{
					{Name: "prompt-strict", Value: report.PromptStrict, Direction: runrecord.DirectionMaximize},
					{Name: "instruction-strict", Value: report.InstructionStrict, Direction: runrecord.DirectionMaximize},
					{Name: "prompt-loose", Value: report.PromptLoose, Direction: runrecord.DirectionMaximize},
					{Name: "instruction-loose", Value: report.InstructionLoose, Direction: runrecord.DirectionMaximize},
				}}, err
			})
	case "":
		envelope.Kind = exactGenerationKind
		return compileKind(data, envelope, authorities, CompileExact, BindExact, []runrecord.Metric{{
			Name: exactMetricName, Direction: runrecord.DirectionMaximize,
		}},
			func(ctx context.Context, repository artifact.Repository, runtime Runtime, compiled ExactPlan, plan Plan) (suiteResult, error) {
				report, err := EvaluateExactSharded(ctx, repository, runtime, compiled, plan, nil)
				return suiteResult{Report: report, Metrics: []runrecord.Metric{{
					Name: exactMetricName, Value: 1, Direction: runrecord.DirectionMaximize,
				}}}, err
			})
	default:
		return CompiledSuite{}, fmt.Errorf("evaluation: unknown suite kind %q", envelope.Kind)
	}
}

// compileKind decodes a suite document of one kind, compiles it, binds its plan and runs it through evaluate.
//   - descriptor kind, source and case count = the envelope's
//   - a kind whose metric contract reads its compiled plan or its decoded cases keeps its own case of compileSuite, because metrics come before the compile here
func compileKind[S, C any](
	data []byte, envelope suiteEnvelope, authorities ExactAuthorities,
	compile func(S) (C, error), bind func(C, ExactAuthorities) (Plan, error), metrics []runrecord.Metric,
	evaluate func(context.Context, artifact.Repository, Runtime, C, Plan) (suiteResult, error),
) (CompiledSuite, error) {
	var source S
	if err := strictjson.DecodeBytes(data, &source); err != nil {
		return CompiledSuite{}, err
	}
	compiled, err := compile(source)
	if err != nil {
		return CompiledSuite{}, err
	}
	plan, err := bind(compiled, authorities)
	return newCompiledSuite(envelope.Kind, envelope.Source, uint64(len(envelope.Cases)), plan, err, metrics,
		func(ctx context.Context, repository artifact.Repository, runtime Runtime) (suiteResult, error) {
			return evaluate(ctx, repository, runtime, compiled, plan)
		})
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

// suiteIdentities identifies a suite, the dataset of its cases and the split that holds that whole dataset.
//   - identity = the profile identity of the suite
//   - dataset = the dataset identity of cases, which a kind writes as its cases or as its cases with how they are counted
//   - split = the dataset shard identity of the dataset alone
//   - stored plans, campaigns and answers name these identities, so their documents never change
//   - an error leaves the identities after it unset, and a caller returns no plan with it
func suiteIdentities(suite, cases any) (identity, dataset, split artifact.ID, err error) {
	identity, err = artifact.JSONID(artifact.KindProfile, suite)
	if err == nil {
		dataset, err = artifact.JSONID(artifact.KindDataset, cases)
	}
	if err == nil {
		split, err = artifact.JSONID(artifact.KindDatasetShard, struct {
			Dataset artifact.ID `json:"dataset"`
		}{Dataset: dataset})
	}
	return identity, dataset, split, err
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
