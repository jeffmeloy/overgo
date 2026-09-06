package evaluation

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
)

const (
	transcriptionResourceReportSchema   = "overgo/transcription-resource-report/v2"
	transcriptionLoadReceiptSchema      = "overgo/transcription-load-receipt/v2"
	transcriptionExecutionReceiptSchema = "overgo/transcription-execution-receipt/v1"
)

var (
	transcriptionExecutionReceiptContract = artifact.JSONContract(artifact.KindEvidence, transcriptionExecutionReceiptSchema)
	transcriptionResourceReportContract   = artifact.JSONContract(
		artifact.KindEvaluation, transcriptionResourceReportSchema,
	)
	transcriptionLoadReceiptContract = artifact.JSONContract(
		artifact.KindEvidence, transcriptionLoadReceiptSchema,
	)
)

// TranscriptionResourceInput is the target-free execution input for one suite
// case. Audio bytes are never copied into the persisted report.
type TranscriptionResourceInput struct {
	Name      string
	Reference dataset.AudioPayloadReference
	Policy    dataset.AudioInspectionPolicy
}

// TranscriptionResourceOptions bounds warmup and measured repetitions.
type TranscriptionResourceOptions struct {
	WarmupRuns uint32 `json:"warmup_runs"`
	TimedRuns  uint32 `json:"timed_runs"`
}

// TranscriptionResourceMeasurement records one cold load or one execution.
// WallNS and ProcessAllocations cover the entire call, including publication.
// Allocations are process-wide, not isolated to this goroutine. RecordedWorkNS
// is the inner run span; UnrecordedNS is the remainder, including final output
// publication and call overhead, not an isolated publication measurement.
// Process peak memory and unobserved interaction counters remain unavailable.
// Observation identifies the canonical runrecord resource summary.
type TranscriptionResourceMeasurement struct {
	Name               string            `json:"name"`
	Repetition         uint32            `json:"repetition"`
	Warmup             bool              `json:"warmup,omitzero"`
	Attempt            artifact.ID       `json:"attempt,omitzero"`
	Observation        artifact.ID       `json:"observation,omitzero"`
	Run                artifact.ID       `json:"run,omitzero"`
	Output             artifact.ID       `json:"output,omitzero"`
	Outcome            runrecord.Outcome `json:"outcome"`
	Failure            string            `json:"failure,omitzero"`
	WallNS             uint64            `json:"wall_ns"`
	RecordedWorkNS     uint64            `json:"recorded_work_ns,omitzero"`
	UnrecordedNS       uint64            `json:"unrecorded_ns,omitzero"`
	ProcessAllocations uint64            `json:"process_allocations"`
	SampleCount        uint64            `json:"sample_count,omitzero"`
	SampleRate         uint64            `json:"sample_rate,omitzero"`
	SilentControl      bool              `json:"silent_control,omitzero"`
	RealTimeFactor     float64           `json:"real_time_factor,omitzero"`
}

// TranscriptionResourceSummary aggregates timed repetitions only. Cold load
// and warmup observations remain separate measurements.
type TranscriptionResourceSummary struct {
	Runs                uint64  `json:"runs"`
	AudioSeconds        float64 `json:"audio_seconds"`
	WallSeconds         float64 `json:"wall_seconds"`
	RecordedWorkSeconds float64 `json:"recorded_work_seconds"`
	UnrecordedSeconds   float64 `json:"unrecorded_seconds"`
	RealTimeFactor      float64 `json:"real_time_factor"`
	ProcessAllocations  uint64  `json:"process_allocations"`
	AdmissionFailures   uint64  `json:"admission_failures"`
	InferenceFailures   uint64  `json:"inference_failures"`
}

// TranscriptionResourceReport joins freshly executed resource evidence to the
// ordinary target-isolated transcription quality report.
type TranscriptionResourceReport struct {
	ID           artifact.ID                        `json:"-"`
	Version      uint16                             `json:"version"`
	Plan         artifact.ID                        `json:"plan"`
	Model        artifact.ID                        `json:"model"`
	Dataset      artifact.ID                        `json:"dataset"`
	Split        artifact.ID                        `json:"split"`
	Options      TranscriptionResourceOptions       `json:"options"`
	Load         TranscriptionResourceMeasurement   `json:"load"`
	Executions   []TranscriptionResourceMeasurement `json:"executions"`
	Summary      TranscriptionResourceSummary       `json:"summary"`
	Quality      artifact.ID                        `json:"quality"`
	QualityScore TranscriptionSlice                 `json:"quality_score"`
	Metrics      []runrecord.Metric                 `json:"metrics"`
}

type transcriptionLoadReceipt struct {
	Version            uint16            `json:"version"`
	Plan               artifact.ID       `json:"plan"`
	Model              artifact.ID       `json:"model"`
	Outcome            runrecord.Outcome `json:"outcome"`
	Failure            string            `json:"failure,omitzero"`
	WallNS             uint64            `json:"wall_ns"`
	ProcessAllocations uint64            `json:"process_allocations"`
}

type transcriptionExecutionSignature struct {
	outcome runrecord.Outcome
	failure string
	output  artifact.ID
}

type transcriptionMemoryBoundary struct {
	mallocs uint64
}

// EvaluateTranscriptionResources measures the native cold load once, then invokes the
// loaded recipe for every warmup and timed case. Quality is scored from the
// first timed execution; no stored transcript is counted as throughput.
func EvaluateTranscriptionResources(
	ctx context.Context,
	repository artifact.Repository,
	compiled TranscriptionPlan,
	plan Plan,
	model artifact.ID,
	inputs []TranscriptionResourceInput,
	options TranscriptionResourceOptions,
	memoryBytes uint64,
) (TranscriptionResourceReport, error) {
	if ctx == nil || repository == nil || memoryBytes == 0 || model.Kind() != artifact.KindModel ||
		compiled.identity != plan.body.CaseProfile ||
		compiled.dataset != plan.body.Dataset || compiled.split != plan.body.Split || options.TimedRuns == 0 {
		return TranscriptionResourceReport{}, errors.New("evaluation: invalid transcription resource request")
	}
	if err := plan.ValidateIdentity(); err != nil {
		return TranscriptionResourceReport{}, err
	}
	if err := requireEvaluationModelDefinition(ctx, repository, plan.body.ModelDefinition, model); err != nil {
		return TranscriptionResourceReport{}, err
	}
	definition, err := recipe.RequireDefinition(ctx, repository, plan.body.RuntimeRecipe)
	if err != nil {
		return TranscriptionResourceReport{}, err
	}
	boundModel, found := definition.PrimaryDependency(recipe.DependencyModel)
	if !found || boundModel != model {
		return TranscriptionResourceReport{}, errors.New("evaluation: transcription recipe model differs")
	}
	orderedInputs, err := bindTranscriptionResourceInputs(compiled, inputs)
	if err != nil {
		return TranscriptionResourceReport{}, err
	}
	source, err := dataset.NewAudioPayloadReader(memoryBytes)
	if err != nil {
		return TranscriptionResourceReport{}, err
	}
	defer source.Close()
	totalRounds := uint64(options.WarmupRuns) + uint64(options.TimedRuns)
	if totalRounds > uint64(math.MaxInt)/uint64(len(orderedInputs)) {
		return TranscriptionResourceReport{}, errors.New("evaluation: transcription resource request is too large")
	}
	if err = publishPlanAuthorities(ctx, repository, plan, nil); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return TranscriptionResourceReport{}, err
	}
	loadBefore := readTranscriptionMemoryBoundary()
	loadStarted := time.Now()
	executor, loadErr := speechrecognition.LoadTranscriber(ctx, repository, plan.body.RuntimeRecipe, memoryBytes)
	loadWall := elapsedResourceNanoseconds(loadStarted)
	loadAfter := readTranscriptionMemoryBoundary()
	if loadErr != nil {
		return TranscriptionResourceReport{}, loadErr
	}
	report := TranscriptionResourceReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Model: model,
		Dataset: compiled.dataset, Split: compiled.split, Options: options,
	}
	report.Load, err = publishTranscriptionLoadMeasurement(
		ctx, repository, plan, model, loadWall, loadBefore, loadAfter,
	)
	if err != nil {
		return TranscriptionResourceReport{}, err
	}
	report.Executions = make([]TranscriptionResourceMeasurement, 0, int(totalRounds)*len(orderedInputs))
	predictions := make([]TranscriptionPrediction, len(orderedInputs))
	signatures := make(map[string]transcriptionExecutionSignature, len(orderedInputs))
	workspace := &speechrecognition.TranscriptionWorkspace{}
	attempts := make(map[string]bool)
	for round := uint64(0); round < totalRounds; round++ {
		warmup := round < uint64(options.WarmupRuns)
		repetition := uint32(round)
		if !warmup {
			repetition = uint32(round - uint64(options.WarmupRuns))
		}
		for index, input := range orderedInputs {
			measurement, run, signature, executeErr := executeTranscriptionResourceCase(
				ctx, repository, executor, workspace, source, plan, model, compiled.suite.Cases[index], input,
				warmup, repetition, report.Load.Attempt, attempts,
			)
			if executeErr != nil && !run.ID.Valid() {
				return TranscriptionResourceReport{}, executeErr
			}
			if err := recordTranscriptionResourceSignature(signatures, input.Name, signature); err != nil {
				return TranscriptionResourceReport{}, err
			}
			report.Executions = append(report.Executions, measurement)
			if !warmup {
				accumulateTranscriptionResources(&report.Summary, measurement)
				if repetition == 0 {
					predictions[index] = TranscriptionPrediction{Name: input.Name, Run: run.ID}
				}
			}
		}
	}
	quality, err := EvaluateTranscription(ctx, repository, compiled, plan, predictions)
	if err != nil {
		return TranscriptionResourceReport{}, err
	}
	report.Quality, report.QualityScore = quality.ID, quality.Overall
	if report.Summary.AudioSeconds > 0 {
		report.Summary.RealTimeFactor = report.Summary.WallSeconds / report.Summary.AudioSeconds
	}
	report.Metrics = transcriptionResourceMetrics(report)
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return TranscriptionResourceReport{}, err
	}
	report.ID = id
	if err = publishTranscriptionResourceReport(ctx, repository, report); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return TranscriptionResourceReport{}, err
	}
	return report, nil
}

func bindTranscriptionResourceInputs(compiled TranscriptionPlan, inputs []TranscriptionResourceInput) ([]TranscriptionResourceInput, error) {
	if len(inputs) != len(compiled.suite.Cases) {
		return nil, errors.New("evaluation: transcription resource inputs differ from suite")
	}
	inputs = slices.Clone(inputs)
	slices.SortFunc(inputs, func(left, right TranscriptionResourceInput) int { return cmp.Compare(left.Name, right.Name) })
	for index, input := range inputs {
		testCase := compiled.suite.Cases[index]
		if input.Name != testCase.Name || input.Reference.Path == "" || input.Reference.Audio != testCase.Source.Audio ||
			input.Reference.Origin.Validate() != nil ||
			input.Policy.Validate() != nil || index > 0 && inputs[index-1].Name == input.Name {
			return nil, errors.New("evaluation: invalid transcription resource input")
		}
		if row := input.Reference.Origin.Row; row != nil {
			inputs[index].Reference.Origin.Row = new(*row)
		}
	}
	return inputs, nil
}

func executeTranscriptionResourceCase(
	ctx context.Context,
	repository artifact.Repository,
	executor *speechrecognition.Transcriber,
	workspace *speechrecognition.TranscriptionWorkspace,
	source *dataset.AudioPayloadReader,
	plan Plan,
	model artifact.ID,
	testCase TranscriptionCase,
	input TranscriptionResourceInput,
	warmup bool,
	repetition uint32,
	loadAttempt artifact.ID,
	attempts map[string]bool,
) (TranscriptionResourceMeasurement, runrecord.Run, transcriptionExecutionSignature, error) {
	// Container I/O is outside the inference timing boundary, as it was when
	// callers eagerly loaded input. Only this attempt's encoded payload is live.
	data, err := source.Read(ctx, input.Reference, input.Policy.MaximumEncodedBytes)
	if err != nil {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, err
	}
	before := readTranscriptionMemoryBoundary()
	started := time.Now()
	_, returnedRun, executeErr := executor.Transcribe(ctx, data, input.Reference.Origin, input.Policy, workspace, speechrecognition.RunBinding{
		Key:        fmt.Sprintf("evaluation/transcription-resource/%s/%t/%d/%s", loadAttempt.DigestHex(), warmup, repetition, input.Name),
		CodeCommit: plan.body.CodeCommit, Environment: plan.body.Environment,
		Dataset: plan.body.Dataset, Split: plan.body.Split,
	})
	wallNS := elapsedResourceNanoseconds(started)
	after := readTranscriptionMemoryBoundary()
	if !returnedRun.ID.Valid() {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, errors.Join(errors.New("evaluation: transcription returned no persisted run"), executeErr)
	}
	run, err := runrecord.RequireExactRun(ctx, repository, returnedRun.ID)
	if err != nil {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, err
	}
	if err = validateTranscriptionResourceRun(run, plan, model, testCase); err != nil {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, err
	}
	if run.MeasuredNS > wallNS {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, errors.New("evaluation: recorded work exceeds measured call")
	}
	measurement := TranscriptionResourceMeasurement{
		Name: input.Name, Repetition: repetition, Warmup: warmup, Run: run.ID,
		Outcome: run.Outcome, Failure: run.Failure, WallNS: wallNS,
		RecordedWorkNS: run.MeasuredNS, UnrecordedNS: wallNS - run.MeasuredNS,
		ProcessAllocations: after.mallocs - min(after.mallocs, before.mallocs),
		SampleCount:        testCase.SampleCount, SampleRate: testCase.SampleRate,
		SilentControl: testCase.SilentControl,
	}
	durationNS := float64(testCase.SampleCount) * float64(time.Second) / float64(testCase.SampleRate)
	if durationNS > 0 {
		measurement.RealTimeFactor = float64(measurement.WallNS) / durationNS
	}
	signature := transcriptionExecutionSignature{outcome: run.Outcome, failure: run.Failure}
	if run.Outcome == runrecord.OutcomeSucceeded {
		output, _, outputErr := requireTranscriptionOutput(ctx, repository, run, testCase.Source)
		if outputErr != nil {
			return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, outputErr
		}
		measurement.Output, signature.output = output, output
	}
	attempt, observation, err := publishTranscriptionExecutionMeasurement(ctx, repository, plan, model, loadAttempt, uint64(len(data)), measurement, attempts)
	if err != nil {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, err
	}
	measurement.Attempt, measurement.Observation = attempt, observation
	return measurement, run, signature, executeErr
}

func validateTranscriptionResourceRun(run runrecord.Run, plan Plan, model artifact.ID, testCase TranscriptionCase) error {
	if run.Recipe != plan.body.RuntimeRecipe || run.CodeCommit != plan.body.CodeCommit || run.Environment != plan.body.Environment ||
		!slices.Contains(run.Inputs, model) || !slices.Contains(run.Inputs, plan.body.Dataset) ||
		!slices.Contains(run.Inputs, plan.body.Split) || !slices.Contains(run.Inputs, testCase.Source.Audio) ||
		!slices.Contains(run.Inputs, testCase.Source.Profile) {
		return errors.New("evaluation: transcription resource run authority differs")
	}
	if !run.ID.Valid() {
		return errors.New("evaluation: transcription resource run is absent")
	}
	return nil
}

func recordTranscriptionResourceSignature(signatures map[string]transcriptionExecutionSignature, name string, signature transcriptionExecutionSignature) error {
	if prior, found := signatures[name]; found && prior != signature {
		return fmt.Errorf("evaluation: transcription output changed for %s", name)
	}
	signatures[name] = signature
	return nil
}

func publishTranscriptionLoadMeasurement(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	model artifact.ID,
	wallNS uint64,
	before, after transcriptionMemoryBoundary,
) (TranscriptionResourceMeasurement, error) {
	receipt := transcriptionLoadReceipt{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Model: model,
		Outcome: runrecord.OutcomeSucceeded, WallNS: wallNS,
		ProcessAllocations: after.mallocs - min(after.mallocs, before.mallocs),
	}
	content, err := artifact.JSONContent(transcriptionLoadReceiptContract, receipt)
	if err != nil {
		return TranscriptionResourceMeasurement{}, err
	}
	fitness, err := transcriptionResourceFitness(plan, model, content.Descriptor.ID, wallNS)
	if err != nil {
		return TranscriptionResourceMeasurement{}, err
	}
	chunk, err := runrecord.NewInitialObservationChunk(fitness, runrecord.ObservationSampleExecution, wallNS)
	if err != nil {
		return TranscriptionResourceMeasurement{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/transcription-resource/load/"+content.Descriptor.ID.DigestHex(), []artifact.Content{content},
		artifact.DependencyLineage(content.Descriptor.ID, plan.identity, model, plan.body.ModelDefinition, plan.body.RuntimeRecipe, plan.body.Environment), nil,
	)
	if err != nil {
		return TranscriptionResourceMeasurement{}, err
	}
	summary, err := runrecord.BindObservationChunk(ctx, repository, &batch, chunk)
	if err != nil {
		return TranscriptionResourceMeasurement{}, err
	}
	if _, err = artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return TranscriptionResourceMeasurement{}, err
	}
	return TranscriptionResourceMeasurement{
		Name: "cold-load", Attempt: content.Descriptor.ID, Observation: summary.ID,
		Outcome: receipt.Outcome, WallNS: wallNS, ProcessAllocations: receipt.ProcessAllocations,
	}, nil
}

func publishTranscriptionExecutionMeasurement(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	model, loadAttempt artifact.ID,
	inputBytes uint64,
	measurement TranscriptionResourceMeasurement,
	attempts map[string]bool,
) (artifact.ID, artifact.ID, error) {
	// Equal content-addressed runs are legitimate. A repetition is identified by
	// its load, case and position, plus the observed result and full-call measures.
	receipt, err := artifact.JSONContent(transcriptionExecutionReceiptContract, struct {
		Plan        artifact.ID                      `json:"plan"`
		Model       artifact.ID                      `json:"model"`
		Load        artifact.ID                      `json:"load"`
		Measurement TranscriptionResourceMeasurement `json:"measurement"`
	}{plan.identity, model, loadAttempt, measurement})
	if err != nil {
		return artifact.ID{}, artifact.ID{}, err
	}
	attempt := receipt.Descriptor.ID
	position := fmt.Sprintf("%t/%d/%s", measurement.Warmup, measurement.Repetition, measurement.Name)
	if attempts[position] {
		return artifact.ID{}, artifact.ID{}, errors.New("evaluation: transcription resource repetition is repeated")
	}
	fitness, err := transcriptionResourceFitness(plan, model, attempt, measurement.WallNS,
		runrecord.ResourceMeasure{Metric: runrecord.ResourceInputBytes, Value: inputBytes})
	if err != nil {
		return artifact.ID{}, artifact.ID{}, err
	}
	chunk, err := runrecord.NewInitialObservationChunk(fitness, runrecord.ObservationSampleExecution, measurement.WallNS)
	if err != nil {
		return artifact.ID{}, artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch("evaluation/transcription-resource/observation/"+attempt.DigestHex(),
		[]artifact.Content{receipt}, artifact.DependencyLineage(attempt, plan.identity, model, loadAttempt, measurement.Run), nil)
	if err != nil {
		return artifact.ID{}, artifact.ID{}, err
	}
	summary, err := runrecord.BindObservationChunk(ctx, repository, &batch, chunk)
	if err != nil {
		return artifact.ID{}, artifact.ID{}, err
	}
	if _, err = artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return artifact.ID{}, artifact.ID{}, err
	}
	attempts[position] = true
	return attempt, summary.ID, nil
}

func transcriptionResourceFitness(
	plan Plan,
	model, attempt artifact.ID,
	wallNS uint64,
	extraMeasures ...runrecord.ResourceMeasure,
) (runrecord.ResourceFitness, error) {
	measures := []runrecord.ResourceMeasure{
		{Metric: runrecord.ResourceWallNS, Value: wallNS},
	}
	measures = append(measures, extraMeasures...)
	return runrecord.NewResourceFitness(runrecord.ResourceFitness{
		Scope: runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation, Model: model, Hardware: plan.body.Environment,
			Workload: plan.identity, Attempt: attempt,
		},
		Measures: measures,
	})
}

func accumulateTranscriptionResources(summary *TranscriptionResourceSummary, measurement TranscriptionResourceMeasurement) {
	summary.Runs++
	summary.AudioSeconds += float64(measurement.SampleCount) / float64(measurement.SampleRate)
	summary.WallSeconds += float64(measurement.WallNS) / float64(time.Second)
	summary.ProcessAllocations += measurement.ProcessAllocations
	summary.RecordedWorkSeconds += float64(measurement.RecordedWorkNS) / float64(time.Second)
	summary.UnrecordedSeconds += float64(measurement.UnrecordedNS) / float64(time.Second)
	if !measurement.SilentControl && measurement.Outcome != runrecord.OutcomeSucceeded {
		if measurement.Failure == speechrecognition.AudioAdmissionFailure {
			summary.AdmissionFailures++
		} else {
			summary.InferenceFailures++
		}
	}
}

func transcriptionResourceMetrics(report TranscriptionResourceReport) []runrecord.Metric {
	metrics := []runrecord.Metric{
		{Name: "process-allocations", Value: float64(report.Summary.ProcessAllocations), Unit: "count", Direction: runrecord.DirectionMinimize},
		{Name: "audio-duration", Value: report.Summary.AudioSeconds, Unit: "seconds", Direction: runrecord.DirectionNeutral},
		{Name: "cold-load-process-allocations", Value: float64(report.Load.ProcessAllocations), Unit: "count", Direction: runrecord.DirectionMinimize},
		{Name: "cold-load-wall", Value: float64(report.Load.WallNS) / float64(time.Second), Unit: "seconds", Direction: runrecord.DirectionMinimize},
		{Name: "real-time-factor", Value: report.Summary.RealTimeFactor, Unit: "ratio", Direction: runrecord.DirectionMinimize},
		{Name: "timed-runs", Value: float64(report.Summary.Runs), Unit: "count", Direction: runrecord.DirectionNeutral},
		{Name: "recorded-work", Value: report.Summary.RecordedWorkSeconds, Unit: "seconds", Direction: runrecord.DirectionMinimize},
		{Name: "unrecorded-tail", Value: report.Summary.UnrecordedSeconds, Unit: "seconds", Direction: runrecord.DirectionNeutral},
		{Name: "wall", Value: report.Summary.WallSeconds, Unit: "seconds", Direction: runrecord.DirectionMinimize},
	}
	for _, metric := range reportQualityMetrics(report.QualityScore) {
		metrics = append(metrics, metric)
	}
	slices.SortFunc(metrics, func(left, right runrecord.Metric) int { return cmp.Compare(left.Name, right.Name) })
	return metrics
}

func reportQualityMetrics(quality TranscriptionSlice) []runrecord.Metric {
	return []runrecord.Metric{
		{Name: "quality-admission-failures", Value: float64(quality.AdmissionFailures), Unit: "count", Direction: runrecord.DirectionMinimize},
		{Name: "quality-cer", Value: quality.CharacterErrorRate, Unit: "ratio", Direction: runrecord.DirectionMinimize},
		{Name: "quality-inference-failures", Value: float64(quality.InferenceFailures), Unit: "count", Direction: runrecord.DirectionMinimize},
		{Name: "quality-wer", Value: quality.WordErrorRate, Unit: "ratio", Direction: runrecord.DirectionMinimize},
	}
}

func publishTranscriptionResourceReport(ctx context.Context, repository artifact.Repository, report TranscriptionResourceReport) error {
	content, err := transcriptionResourceReportContract.ContentJSON(report.ID, report)
	if err != nil {
		return err
	}
	parents := []artifact.ID{report.Plan, report.Model, report.Dataset, report.Split, report.Load.Attempt, report.Load.Observation, report.Quality}
	for _, execution := range report.Executions {
		parents = append(parents, execution.Run, execution.Attempt, execution.Observation)
		if execution.Output.Valid() {
			parents = append(parents, execution.Output)
		}
	}
	parents = uniqueArtifactIDs(parents)
	alias := "evaluation/transcription-resource/report/" + report.Plan.String()
	return publishTranscriptionReportContent(ctx, repository, alias, content, parents)
}

func readTranscriptionMemoryBoundary() transcriptionMemoryBoundary {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return transcriptionMemoryBoundary{mallocs: stats.Mallocs}
}

func elapsedResourceNanoseconds(started time.Time) uint64 {
	elapsed := time.Since(started).Nanoseconds()
	if elapsed <= 0 {
		return uint64(time.Nanosecond)
	}
	return uint64(elapsed)
}
