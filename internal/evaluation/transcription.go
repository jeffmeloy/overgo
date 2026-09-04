package evaluation

import (
	"cmp"
	"context"
	"errors"
	"maps"
	"math"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/textcheck"
)

const (
	// TranscriptionKind identifies a target-isolated speech-recognition suite.
	TranscriptionKind = "transcription"

	transcriptionReportMedia  = "application/vnd.overgo.transcription-report+json"
	transcriptionReportSchema = "overgo/transcription-report/v1"
)

var transcriptionReportContract = artifact.DocumentContract{
	Kind: artifact.KindEvaluation, MediaType: transcriptionReportMedia, Schema: transcriptionReportSchema,
}

var transcriptionNormalizations = []TranscriptionNormalization{
	TranscriptionLowercase, TranscriptionStripPunctuation, TranscriptionCollapseWhitespace,
}

// TranscriptionNormalization names one ordered, code-owned text transform.
// The ordered list is part of the evaluator plan identity.
type TranscriptionNormalization string

const (
	// TranscriptionLowercase applies Unicode lowercase mapping.
	TranscriptionLowercase TranscriptionNormalization = "lowercase"
	// TranscriptionStripPunctuation removes Unicode punctuation runes.
	TranscriptionStripPunctuation TranscriptionNormalization = "strip-punctuation"
	// TranscriptionCollapseWhitespace trims and collapses Unicode whitespace.
	TranscriptionCollapseWhitespace TranscriptionNormalization = "collapse-whitespace"
)

// TranscriptionCase binds a reference only to an already pinned audio source.
// Silent controls carry no reference and must be refused by admission.
type TranscriptionCase struct {
	Name          string                        `json:"name"`
	Group         string                        `json:"group"`
	Source        recipecontract.AudioReference `json:"source"`
	Reference     string                        `json:"reference,omitzero"`
	SampleCount   uint64                        `json:"sample_count"`
	SampleRate    uint64                        `json:"sample_rate"`
	SilentControl bool                          `json:"silent_control,omitzero"`
}

// TranscriptionSuite declares pinned source, split, targets, and scoring
// protocol. It contains no prediction or execution result.
type TranscriptionSuite struct {
	Kind          string                       `json:"kind"`
	Schema        string                       `json:"schema"`
	Source        string                       `json:"source"`
	Dataset       artifact.ID                  `json:"dataset"`
	Split         artifact.ID                  `json:"split"`
	Normalization []TranscriptionNormalization `json:"normalization"`
	Cases         []TranscriptionCase          `json:"cases"`
}

// TranscriptionPlan is the compiled, target-bearing portion of evaluation.
// Prediction code receives TranscriptionPrediction values, not this type.
type TranscriptionPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    TranscriptionSuite
}

// TranscriptionPrediction is the target-free handoff from inference to
// scoring. The run resolves the raw output and all execution authorities.
type TranscriptionPrediction struct {
	Name string      `json:"name"`
	Run  artifact.ID `json:"run"`
}

// TranscriptionObservation records one exact prediction outcome and its error
// counts. Raw text remains in the referenced output artifact.
type TranscriptionObservation struct {
	Name             string            `json:"name"`
	Group            string            `json:"group"`
	Source           artifact.ID       `json:"source"`
	Run              artifact.ID       `json:"run"`
	Output           artifact.ID       `json:"output,omitzero"`
	Outcome          runrecord.Outcome `json:"outcome"`
	Failure          string            `json:"failure,omitzero"`
	CodeCommit       string            `json:"code_commit"`
	Environment      artifact.ID       `json:"environment"`
	WordEdits        uint64            `json:"word_edits"`
	ReferenceWords   uint64            `json:"reference_words"`
	CharacterEdits   uint64            `json:"character_edits"`
	ReferenceRunes   uint64            `json:"reference_runes"`
	SampleCount      uint64            `json:"sample_count"`
	SampleRate       uint64            `json:"sample_rate"`
	SilentControl    bool              `json:"silent_control,omitzero"`
	ControlViolation bool              `json:"control_violation,omitzero"`
}

// TranscriptionSlice reports micro-averaged quality and complete failure
// denominators for the full suite or one declared group.
type TranscriptionSlice struct {
	Name                  string  `json:"name"`
	WordErrorRate         float64 `json:"word_error_rate"`
	CharacterErrorRate    float64 `json:"character_error_rate"`
	Utterances            uint64  `json:"utterances"`
	SampleCount           uint64  `json:"sample_count"`
	DurationSeconds       float64 `json:"duration_seconds"`
	AdmissionFailures     uint64  `json:"admission_failures"`
	InferenceFailures     uint64  `json:"inference_failures"`
	SilentControls        uint64  `json:"silent_controls"`
	SilentControlFailures uint64  `json:"silent_control_failures"`
}

// TranscriptionReport is the immutable report derived without invoking model
// inference. Metrics mirror Overall and Groups through runrecord's vocabulary.
type TranscriptionReport struct {
	ID            artifact.ID                  `json:"-"`
	Version       uint16                       `json:"version"`
	Plan          artifact.ID                  `json:"plan"`
	Dataset       artifact.ID                  `json:"dataset"`
	Split         artifact.ID                  `json:"split"`
	Normalization []TranscriptionNormalization `json:"normalization"`
	Observations  []TranscriptionObservation   `json:"observations"`
	Overall       TranscriptionSlice           `json:"overall"`
	Groups        []TranscriptionSlice         `json:"groups"`
	Metrics       []runrecord.Metric           `json:"metrics"`
}

type transcriptionAccumulator struct {
	name                                  string
	wordEdits, referenceWords             uint64
	characterEdits, referenceRunes        uint64
	utterances, sampleCount               uint64
	admissionFailures, inferenceFailures  uint64
	silentControls, silentControlFailures uint64
	durationSeconds                       float64
}

// CompileTranscription validates and identifies a pinned transcription suite.
func CompileTranscription(suite TranscriptionSuite) (TranscriptionPlan, error) {
	if suite.Kind != TranscriptionKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || suite.Dataset.Kind() != artifact.KindDataset ||
		suite.Split.Kind() != artifact.KindDatasetShard || len(suite.Normalization) == 0 || len(suite.Cases) == 0 {
		return TranscriptionPlan{}, errors.New("evaluation: invalid transcription suite")
	}
	suite.Normalization = slices.Clone(suite.Normalization)
	seenOperations := make(map[TranscriptionNormalization]struct{}, len(suite.Normalization))
	for _, operation := range suite.Normalization {
		if !slices.Contains(transcriptionNormalizations, operation) {
			return TranscriptionPlan{}, errors.New("evaluation: unknown transcription normalization")
		}
		if _, duplicate := seenOperations[operation]; duplicate {
			return TranscriptionPlan{}, errors.New("evaluation: duplicate transcription normalization")
		}
		seenOperations[operation] = struct{}{}
	}
	suite.Cases = slices.Clone(suite.Cases)
	var totalSamples uint64
	groupLabels := make(map[string]string)
	for index := range suite.Cases {
		if err := validateTranscriptionCase(suite.Cases[index], suite.Normalization); err != nil {
			return TranscriptionPlan{}, err
		}
		if totalSamples > math.MaxUint64-suite.Cases[index].SampleCount {
			return TranscriptionPlan{}, errors.New("evaluation: transcription sample count overflows")
		}
		totalSamples += suite.Cases[index].SampleCount
		label := metricLabel(suite.Cases[index].Group)
		if prior, exists := groupLabels[label]; exists && prior != suite.Cases[index].Group {
			return TranscriptionPlan{}, errors.New("evaluation: transcription group metric labels collide")
		}
		groupLabels[label] = suite.Cases[index].Group
	}
	slices.SortFunc(suite.Cases, func(left, right TranscriptionCase) int { return cmp.Compare(left.Name, right.Name) })
	for index := 1; index < len(suite.Cases); index++ {
		if suite.Cases[index-1].Name == suite.Cases[index].Name {
			return TranscriptionPlan{}, errors.New("evaluation: duplicate transcription case")
		}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return TranscriptionPlan{}, err
	}
	return TranscriptionPlan{identity: identity, dataset: suite.Dataset, split: suite.Split, suite: suite}, nil
}

// BindTranscription binds model, recipe, code, and environment authorities to
// the suite using the common evaluation plan.
func BindTranscription(compiled TranscriptionPlan, authorities ExactAuthorities) (Plan, error) {
	if !compiled.identity.Valid() {
		return Plan{}, errors.New("evaluation: transcription suite is not compiled")
	}
	return bindPlan(
		compiled.dataset, compiled.split, compiled.identity, compiled.suite,
		transcriptionScorer(compiled.suite.Normalization), authorities,
	)
}

func transcriptionScorer(normalization []TranscriptionNormalization) struct {
	Version       uint16                       `json:"version"`
	Kind          string                       `json:"kind"`
	Normalization []TranscriptionNormalization `json:"normalization"`
} {
	return struct {
		Version       uint16                       `json:"version"`
		Kind          string                       `json:"kind"`
		Normalization []TranscriptionNormalization `json:"normalization"`
	}{
		Version: artifact.InitialDocumentVersion, Kind: TranscriptionKind,
		Normalization: slices.Clone(normalization),
	}
}

// EvaluateTranscription scores already persisted target-free predictions. It
// never invokes inference and rejects any run whose exact authorities differ
// from the bound plan.
func EvaluateTranscription(
	ctx context.Context,
	repository artifact.Repository,
	compiled TranscriptionPlan,
	plan Plan,
	predictions []TranscriptionPrediction,
) (TranscriptionReport, error) {
	if ctx == nil || repository == nil || compiled.identity != plan.body.CaseProfile ||
		compiled.dataset != plan.body.Dataset || compiled.split != plan.body.Split ||
		len(predictions) != len(compiled.suite.Cases) {
		return TranscriptionReport{}, errors.New("evaluation: transcription authorities differ")
	}
	scorer, err := authorityContent(scorerProfileContract, transcriptionScorer(compiled.suite.Normalization))
	if err != nil || plan.ValidateIdentity() != nil || scorer.Descriptor.ID != plan.body.Scorer {
		return TranscriptionReport{}, errors.Join(err, errors.New("evaluation: transcription scoring protocol differs"))
	}
	if err := publishPlanAuthorities(ctx, repository, plan, nil); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return TranscriptionReport{}, err
	}
	predictions = slices.Clone(predictions)
	slices.SortFunc(predictions, func(left, right TranscriptionPrediction) int { return cmp.Compare(left.Name, right.Name) })
	report := TranscriptionReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset, Split: compiled.split,
		Normalization: slices.Clone(compiled.suite.Normalization),
		Observations:  make([]TranscriptionObservation, len(predictions)),
	}
	overall := transcriptionAccumulator{name: "overall"}
	groups := make(map[string]*transcriptionAccumulator)
	for index, testCase := range compiled.suite.Cases {
		prediction := predictions[index]
		if prediction.Name != testCase.Name || prediction.Run.Kind() != artifact.KindRun {
			return TranscriptionReport{}, errors.New("evaluation: transcription prediction differs from suite")
		}
		observation, err := scoreTranscriptionPrediction(
			ctx, repository, plan, testCase, prediction, compiled.suite.Normalization,
		)
		if err != nil {
			return TranscriptionReport{}, err
		}
		report.Observations[index] = observation
		accumulateTranscription(&overall, observation)
		group := groups[testCase.Group]
		if group == nil {
			group = &transcriptionAccumulator{name: testCase.Group}
			groups[testCase.Group] = group
		}
		accumulateTranscription(group, observation)
	}
	report.Overall = overall.result()
	groupNames := slices.Sorted(maps.Keys(groups))
	for _, name := range groupNames {
		report.Groups = append(report.Groups, groups[name].result())
	}
	report.Metrics = transcriptionMetrics(report.Overall, report.Groups)
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return TranscriptionReport{}, err
	}
	report.ID = id
	if err = publishTranscriptionReport(ctx, repository, report); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return TranscriptionReport{}, err
	}
	return report, nil
}

func scoreTranscriptionPrediction(
	ctx context.Context,
	repository artifact.Reader,
	plan Plan,
	testCase TranscriptionCase,
	prediction TranscriptionPrediction,
	normalization []TranscriptionNormalization,
) (TranscriptionObservation, error) {
	run, err := runrecord.RequireExactRun(ctx, repository, prediction.Run)
	if err != nil {
		return TranscriptionObservation{}, err
	}
	if run.Recipe != plan.body.RuntimeRecipe || run.CodeCommit != plan.body.CodeCommit ||
		run.Environment != plan.body.Environment || !slices.Contains(run.Inputs, testCase.Source.Audio) ||
		!slices.Contains(run.Inputs, testCase.Source.Profile) || !slices.Contains(run.Inputs, plan.body.Dataset) ||
		!slices.Contains(run.Inputs, plan.body.Split) ||
		run.Outcome != runrecord.OutcomeSucceeded && len(run.Outputs) != 0 {
		return TranscriptionObservation{}, errors.New("evaluation: transcription run authority differs")
	}
	observation := TranscriptionObservation{
		Name: testCase.Name, Group: testCase.Group, Source: testCase.Source.Audio,
		Run: run.ID, Outcome: run.Outcome, Failure: run.Failure, CodeCommit: run.CodeCommit,
		Environment: run.Environment, SampleCount: testCase.SampleCount, SampleRate: testCase.SampleRate,
		SilentControl: testCase.SilentControl,
	}
	if testCase.SilentControl {
		observation.ControlViolation = run.Outcome != runrecord.OutcomeFailed || run.Failure != speechrecognition.AudioAdmissionFailure
		if run.Outcome == runrecord.OutcomeSucceeded {
			output, _, loadErr := requireTranscriptionOutput(ctx, repository, run, testCase.Source)
			if loadErr != nil {
				return TranscriptionObservation{}, loadErr
			}
			observation.Output = output
		}
		return observation, nil
	}
	reference := normalizeTranscription(testCase.Reference, normalization)
	observation.ReferenceWords = uint64(len(strings.Fields(reference)))
	observation.ReferenceRunes = uint64(len([]rune(reference)))
	if run.Outcome != runrecord.OutcomeSucceeded {
		observation.WordEdits = observation.ReferenceWords
		observation.CharacterEdits = observation.ReferenceRunes
		return observation, nil
	}
	output, transcription, err := requireTranscriptionOutput(ctx, repository, run, testCase.Source)
	if err != nil {
		return TranscriptionObservation{}, err
	}
	observation.Output = output
	hypothesis := normalizeTranscription(transcription.Text, normalization)
	observation.WordEdits = uint64(sequenceEditDistance(strings.Fields(reference), strings.Fields(hypothesis)))
	observation.CharacterEdits = uint64(sequenceEditDistance([]rune(reference), []rune(hypothesis)))
	return observation, nil
}

func requireTranscriptionOutput(
	ctx context.Context,
	reader artifact.Reader,
	run runrecord.Run,
	source recipecontract.AudioReference,
) (artifact.ID, recipecontract.Transcription, error) {
	if len(run.Outputs) != 1 || run.Outputs[0].Kind() != artifact.KindOutput {
		return artifact.ID{}, recipecontract.Transcription{}, errors.New("evaluation: transcription run has no singular output")
	}
	transcription, err := speechrecognition.RequireTranscription(ctx, reader, run.Outputs[0])
	if err != nil {
		return artifact.ID{}, recipecontract.Transcription{}, err
	}
	if transcription.Source != source {
		return artifact.ID{}, recipecontract.Transcription{}, errors.New("evaluation: transcription output source differs")
	}
	return run.Outputs[0], transcription, nil
}

func publishTranscriptionReport(ctx context.Context, repository artifact.Repository, report TranscriptionReport) error {
	content, err := transcriptionReportContract.ContentJSON(report.ID, report)
	if err != nil {
		return err
	}
	parents := []artifact.ID{report.Plan, report.Dataset, report.Split}
	for _, observation := range report.Observations {
		parents = append(parents, observation.Run)
		if observation.Output.Valid() {
			parents = append(parents, observation.Output)
		}
	}
	parents = uniqueArtifactIDs(parents)
	alias := campaignAlias(report.Plan)
	batch, err := artifact.NewDocumentBatch(
		alias, []artifact.Content{content}, artifact.DependencyLineage(report.ID, parents...),
		[]artifact.AliasBinding{{Name: alias, Target: report.ID}},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return err
}

func validateTranscriptionCase(testCase TranscriptionCase, normalization []TranscriptionNormalization) error {
	if !textcheck.BoundedToken(testCase.Name, len(testCase.Name), "") ||
		!textcheck.BoundedToken(testCase.Group, len(testCase.Group), "") || testCase.Source.Validate() != nil ||
		testCase.SampleCount == 0 || testCase.SampleRate == 0 ||
		testCase.SampleCount > math.MaxInt64 || testCase.SampleRate > math.MaxInt64 {
		return errors.New("evaluation: invalid transcription case")
	}
	if testCase.SilentControl {
		if testCase.Reference != "" {
			return errors.New("evaluation: silent control carries a reference")
		}
		return nil
	}
	if normalizeTranscription(testCase.Reference, normalization) == "" {
		return errors.New("evaluation: transcription reference is empty after normalization")
	}
	return nil
}

func normalizeTranscription(value string, operations []TranscriptionNormalization) string {
	for _, operation := range operations {
		switch operation {
		case TranscriptionLowercase:
			value = strings.ToLower(value)
		case TranscriptionStripPunctuation:
			value = strings.Map(func(r rune) rune {
				if unicode.IsPunct(r) {
					return -1
				}
				return r
			}, value)
		case TranscriptionCollapseWhitespace:
			value = strings.Join(strings.Fields(value), " ")
		}
	}
	return value
}

func sequenceEditDistance[T comparable](reference, hypothesis []T) int {
	if len(reference) < len(hypothesis) {
		return sequenceEditDistance(hypothesis, reference)
	}
	previous := make([]int, len(hypothesis)+1)
	current := make([]int, len(hypothesis)+1)
	for index := range previous {
		previous[index] = index
	}
	for referenceIndex, referenceValue := range reference {
		current[0] = referenceIndex + 1
		for hypothesisIndex, hypothesisValue := range hypothesis {
			substitution := previous[hypothesisIndex]
			if referenceValue != hypothesisValue {
				substitution++
			}
			current[hypothesisIndex+1] = min(previous[hypothesisIndex+1]+1, current[hypothesisIndex]+1, substitution)
		}
		previous, current = current, previous
	}
	return previous[len(hypothesis)]
}

func accumulateTranscription(accumulator *transcriptionAccumulator, observation TranscriptionObservation) {
	accumulator.sampleCount += observation.SampleCount
	accumulator.durationSeconds += float64(observation.SampleCount) / float64(observation.SampleRate)
	if observation.SilentControl {
		accumulator.silentControls++
		if observation.ControlViolation {
			accumulator.silentControlFailures++
		}
		return
	}
	accumulator.utterances++
	accumulator.wordEdits += observation.WordEdits
	accumulator.referenceWords += observation.ReferenceWords
	accumulator.characterEdits += observation.CharacterEdits
	accumulator.referenceRunes += observation.ReferenceRunes
	if observation.Outcome != runrecord.OutcomeSucceeded {
		if observation.Failure == speechrecognition.AudioAdmissionFailure {
			accumulator.admissionFailures++
		} else {
			accumulator.inferenceFailures++
		}
	}
}

func (accumulator transcriptionAccumulator) result() TranscriptionSlice {
	result := TranscriptionSlice{
		Name: accumulator.name, Utterances: accumulator.utterances, SampleCount: accumulator.sampleCount,
		DurationSeconds: accumulator.durationSeconds, AdmissionFailures: accumulator.admissionFailures,
		InferenceFailures: accumulator.inferenceFailures, SilentControls: accumulator.silentControls,
		SilentControlFailures: accumulator.silentControlFailures,
	}
	if accumulator.referenceWords != 0 {
		result.WordErrorRate = float64(accumulator.wordEdits) / float64(accumulator.referenceWords)
	}
	if accumulator.referenceRunes != 0 {
		result.CharacterErrorRate = float64(accumulator.characterEdits) / float64(accumulator.referenceRunes)
	}
	return result
}

func transcriptionMetrics(overall TranscriptionSlice, groups []TranscriptionSlice) []runrecord.Metric {
	metrics := transcriptionSliceMetrics("", overall)
	for _, group := range groups {
		metrics = append(metrics, transcriptionSliceMetrics("."+metricLabel(group.Name), group)...)
	}
	slices.SortFunc(metrics, func(left, right runrecord.Metric) int { return cmp.Compare(left.Name, right.Name) })
	return metrics
}

func transcriptionSliceMetrics(suffix string, value TranscriptionSlice) []runrecord.Metric {
	return []runrecord.Metric{
		{Name: "admission-failures" + suffix, Value: float64(value.AdmissionFailures), Unit: "count", Direction: runrecord.DirectionNeutral},
		{Name: "cer" + suffix, Value: value.CharacterErrorRate, Unit: "ratio", Direction: runrecord.DirectionMinimize},
		{Name: "duration-seconds" + suffix, Value: value.DurationSeconds, Unit: "seconds", Direction: runrecord.DirectionNeutral},
		{Name: "inference-failures" + suffix, Value: float64(value.InferenceFailures), Unit: "count", Direction: runrecord.DirectionNeutral},
		{Name: "sample-count" + suffix, Value: float64(value.SampleCount), Unit: "samples", Direction: runrecord.DirectionNeutral},
		{Name: "silent-control-failures" + suffix, Value: float64(value.SilentControlFailures), Unit: "count", Direction: runrecord.DirectionMinimize},
		{Name: "silent-controls" + suffix, Value: float64(value.SilentControls), Unit: "count", Direction: runrecord.DirectionNeutral},
		{Name: "utterances" + suffix, Value: float64(value.Utterances), Unit: "count", Direction: runrecord.DirectionNeutral},
		{Name: "wer" + suffix, Value: value.WordErrorRate, Unit: "ratio", Direction: runrecord.DirectionMinimize},
	}
}

func metricLabel(value string) string {
	label := []byte(strings.ToLower(value))
	for index := range label {
		if !textcheck.LowerIdentifierByte(label[index]) {
			label[index] = '-'
		}
	}
	return string(label)
}
