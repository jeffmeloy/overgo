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
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
)

const (
	transcriptionResourceReportSchema = "overgo/transcription-resource-report/v1"
	transcriptionLoadReceiptSchema    = "overgo/transcription-load-receipt/v1"
)

var (
	transcriptionResourceReportContract = artifact.JSONContract(
		artifact.KindEvaluation, transcriptionResourceReportSchema,
	)
	transcriptionLoadReceiptContract = artifact.JSONContract(
		artifact.KindEvidence, transcriptionLoadReceiptSchema,
	)
)

// TranscriptionResourceExecutor is the existing offline transcription
// execution boundary used for every measured repetition.
type TranscriptionResourceExecutor interface {
	Transcribe(context.Context, []byte, dataset.AudioPayloadOrigin, dataset.AudioInspectionPolicy, *speechrecognition.TranscriptionWorkspace, speechrecognition.RunBinding) (recipecontract.Transcription, runrecord.Run, error)
}

// TranscriptionResourceLoader loads one resident offline executor. Its elapsed
// work is recorded separately from warm transcription execution.
type TranscriptionResourceLoader func(context.Context) (TranscriptionResourceExecutor, error)

// TranscriptionResourceInput is the target-free execution input for one suite
// case. Audio bytes are never copied into the persisted report.
type TranscriptionResourceInput struct {
	Name   string
	Data   []byte
	Origin dataset.AudioPayloadOrigin
	Policy dataset.AudioInspectionPolicy
}

// TranscriptionResourceOptions bounds warmup and measured repetitions.
type TranscriptionResourceOptions struct {
	WarmupRuns uint32 `json:"warmup_runs"`
	TimedRuns  uint32 `json:"timed_runs"`
}

// TranscriptionResourceMeasurement records one cold load or one execution.
// Observation identifies the canonical runrecord resource summary.
type TranscriptionResourceMeasurement struct {
	Name           string            `json:"name"`
	Repetition     uint32            `json:"repetition"`
	Warmup         bool              `json:"warmup,omitzero"`
	Attempt        artifact.ID       `json:"attempt"`
	Observation    artifact.ID       `json:"observation"`
	Output         artifact.ID       `json:"output,omitzero"`
	Outcome        runrecord.Outcome `json:"outcome"`
	Failure        string            `json:"failure,omitzero"`
	WallNS         uint64            `json:"wall_ns"`
	Allocations    uint64            `json:"allocations"`
	PeakHostBytes  uint64            `json:"peak_host_bytes"`
	SampleCount    uint64            `json:"sample_count,omitzero"`
	SampleRate     uint64            `json:"sample_rate,omitzero"`
	SilentControl  bool              `json:"silent_control,omitzero"`
	RealTimeFactor float64           `json:"real_time_factor,omitzero"`
}

// TranscriptionResourceSummary aggregates timed repetitions only. Cold load
// and warmup observations remain separate measurements.
type TranscriptionResourceSummary struct {
	Runs              uint64  `json:"runs"`
	AudioSeconds      float64 `json:"audio_seconds"`
	WallSeconds       float64 `json:"wall_seconds"`
	RealTimeFactor    float64 `json:"real_time_factor"`
	Allocations       uint64  `json:"allocations"`
	PeakHostBytes     uint64  `json:"peak_host_bytes"`
	AdmissionFailures uint64  `json:"admission_failures"`
	InferenceFailures uint64  `json:"inference_failures"`
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
	Version       uint16            `json:"version"`
	Plan          artifact.ID       `json:"plan"`
	Model         artifact.ID       `json:"model"`
	Outcome       runrecord.Outcome `json:"outcome"`
	Failure       string            `json:"failure,omitzero"`
	WallNS        uint64            `json:"wall_ns"`
	Allocations   uint64            `json:"allocations"`
	PeakHostBytes uint64            `json:"peak_host_bytes"`
}

type transcriptionExecutionSignature struct {
	outcome runrecord.Outcome
	failure string
	output  artifact.ID
}

type transcriptionMemoryBoundary struct {
	heapAlloc uint64
	mallocs   uint64
}

// EvaluateTranscriptionResources measures a cold loader once, then invokes the
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
	load TranscriptionResourceLoader,
) (TranscriptionResourceReport, error) {
	if ctx == nil || repository == nil || load == nil || model.Kind() != artifact.KindModel ||
		compiled.identity != plan.body.CaseProfile ||
		compiled.dataset != plan.body.Dataset || compiled.split != plan.body.Split || options.TimedRuns == 0 {
		return TranscriptionResourceReport{}, errors.New("evaluation: invalid transcription resource request")
	}
	if err := plan.ValidateIdentity(); err != nil {
		return TranscriptionResourceReport{}, err
	}
	orderedInputs, err := bindTranscriptionResourceInputs(compiled, inputs)
	if err != nil {
		return TranscriptionResourceReport{}, err
	}
	totalRounds := uint64(options.WarmupRuns) + uint64(options.TimedRuns)
	if totalRounds > uint64(math.MaxInt)/uint64(len(orderedInputs)) {
		return TranscriptionResourceReport{}, errors.New("evaluation: transcription resource request is too large")
	}
	if err = publishPlanAuthorities(ctx, repository, plan, nil); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return TranscriptionResourceReport{}, err
	}
	loadBefore := readTranscriptionMemoryBoundary()
	loadStarted := time.Now()
	executor, loadErr := load(ctx)
	loadWall := elapsedResourceNanoseconds(loadStarted)
	loadAfter := readTranscriptionMemoryBoundary()
	if loadErr != nil {
		return TranscriptionResourceReport{}, loadErr
	}
	report := TranscriptionResourceReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Model: model,
		Dataset: compiled.dataset, Split: compiled.split, Options: options,
	}
	report.Executions = make([]TranscriptionResourceMeasurement, 0, int(totalRounds)*len(orderedInputs))
	predictions := make([]TranscriptionPrediction, len(orderedInputs))
	signatures := make(map[string]transcriptionExecutionSignature, len(orderedInputs))
	workspace := &speechrecognition.TranscriptionWorkspace{}
	for round := uint64(0); round < totalRounds; round++ {
		warmup := round < uint64(options.WarmupRuns)
		repetition := uint32(round)
		if !warmup {
			repetition = uint32(round - uint64(options.WarmupRuns))
		}
		for index, input := range orderedInputs {
			measurement, run, signature, executeErr := executeTranscriptionResourceCase(
				ctx, repository, executor, workspace, plan, model, compiled.suite.Cases[index], input,
				warmup, repetition,
			)
			if executeErr != nil && !run.ID.Valid() {
				return TranscriptionResourceReport{}, executeErr
			}
			if prior, found := signatures[input.Name]; found && prior != signature {
				return TranscriptionResourceReport{}, fmt.Errorf("evaluation: transcription output changed for %s", input.Name)
			}
			signatures[input.Name] = signature
			report.Executions = append(report.Executions, measurement)
			if !warmup {
				accumulateTranscriptionResources(&report.Summary, measurement)
				if repetition == 0 {
					predictions[index] = TranscriptionPrediction{Name: input.Name, Run: run.ID}
				}
			}
		}
	}
	report.Load, err = publishTranscriptionLoadMeasurement(
		ctx, repository, plan, model, loadWall, loadBefore, loadAfter,
	)
	if err != nil {
		return TranscriptionResourceReport{}, err
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
		source, err := artifact.IdentifyBytes(artifact.KindFile, input.Data)
		if input.Name != testCase.Name || len(input.Data) == 0 || err != nil || source != testCase.Source.Audio ||
			input.Origin.Container.Kind() != artifact.KindFile && input.Origin.Container.Kind() != artifact.KindDatasetShard ||
			input.Policy.Validate() != nil || index > 0 && inputs[index-1].Name == input.Name {
			return nil, errors.New("evaluation: invalid transcription resource input")
		}
		inputs[index].Data = slices.Clone(input.Data)
	}
	return inputs, nil
}

func executeTranscriptionResourceCase(
	ctx context.Context,
	repository artifact.Repository,
	executor TranscriptionResourceExecutor,
	workspace *speechrecognition.TranscriptionWorkspace,
	plan Plan,
	model artifact.ID,
	testCase TranscriptionCase,
	input TranscriptionResourceInput,
	warmup bool,
	repetition uint32,
) (TranscriptionResourceMeasurement, runrecord.Run, transcriptionExecutionSignature, error) {
	before := readTranscriptionMemoryBoundary()
	_, returnedRun, executeErr := executor.Transcribe(ctx, input.Data, input.Origin, input.Policy, workspace, speechrecognition.RunBinding{
		Key:        fmt.Sprintf("evaluation/transcription-resource/%s/%t/%d/%s", plan.identity.DigestHex(), warmup, repetition, input.Name),
		CodeCommit: plan.body.CodeCommit, Environment: plan.body.Environment,
		Dataset: plan.body.Dataset, Split: plan.body.Split,
	})
	after := readTranscriptionMemoryBoundary()
	if !returnedRun.ID.Valid() {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, executeErr
	}
	run, err := runrecord.RequireExactRun(ctx, repository, returnedRun.ID)
	if err != nil {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, err
	}
	if run.Recipe != plan.body.RuntimeRecipe || run.CodeCommit != plan.body.CodeCommit || run.Environment != plan.body.Environment ||
		!slices.Contains(run.Inputs, model) || !slices.Contains(run.Inputs, plan.body.Dataset) ||
		!slices.Contains(run.Inputs, plan.body.Split) || !slices.Contains(run.Inputs, testCase.Source.Audio) ||
		!slices.Contains(run.Inputs, testCase.Source.Profile) {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, errors.New("evaluation: transcription resource run authority differs")
	}
	measurement := TranscriptionResourceMeasurement{
		Name: input.Name, Repetition: repetition, Warmup: warmup, Attempt: run.ID,
		Outcome: run.Outcome, Failure: run.Failure, WallNS: run.MeasuredNS,
		Allocations:   after.mallocs - min(after.mallocs, before.mallocs),
		PeakHostBytes: max(before.heapAlloc, after.heapAlloc),
		SampleCount:   testCase.SampleCount, SampleRate: testCase.SampleRate,
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
	observation, err := publishTranscriptionExecutionMeasurement(ctx, repository, plan, model, uint64(len(input.Data)), measurement)
	if err != nil {
		return TranscriptionResourceMeasurement{}, runrecord.Run{}, transcriptionExecutionSignature{}, err
	}
	measurement.Observation = observation
	return measurement, run, signature, executeErr
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
		Allocations: after.mallocs - min(after.mallocs, before.mallocs), PeakHostBytes: max(before.heapAlloc, after.heapAlloc),
	}
	content, err := artifact.JSONContent(transcriptionLoadReceiptContract, receipt)
	if err != nil {
		return TranscriptionResourceMeasurement{}, err
	}
	fitness, err := transcriptionResourceFitness(plan, model, content.Descriptor.ID, wallNS, receipt.Allocations, receipt.PeakHostBytes)
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
		Outcome: receipt.Outcome, WallNS: wallNS, Allocations: receipt.Allocations, PeakHostBytes: receipt.PeakHostBytes,
	}, nil
}

func publishTranscriptionExecutionMeasurement(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	model artifact.ID,
	inputBytes uint64,
	measurement TranscriptionResourceMeasurement,
) (artifact.ID, error) {
	fitness, err := transcriptionResourceFitness(
		plan, model, measurement.Attempt, measurement.WallNS, measurement.Allocations,
		measurement.PeakHostBytes, runrecord.ResourceMeasure{Metric: runrecord.ResourceInputBytes, Value: inputBytes},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	chunk, err := runrecord.NewInitialObservationChunk(fitness, runrecord.ObservationSampleExecution, measurement.WallNS)
	if err != nil {
		return artifact.ID{}, err
	}
	batch := artifact.Batch{Key: "evaluation/transcription-resource/observation/" + measurement.Attempt.DigestHex()}
	summary, err := runrecord.BindObservationChunk(ctx, repository, &batch, chunk)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err = artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return artifact.ID{}, err
	}
	return summary.ID, nil
}

func transcriptionResourceFitness(
	plan Plan,
	model, attempt artifact.ID,
	wallNS, allocations, peakHostBytes uint64,
	extraMeasures ...runrecord.ResourceMeasure,
) (runrecord.ResourceFitness, error) {
	measures := []runrecord.ResourceMeasure{
		{Metric: runrecord.ResourceWallNS, Value: wallNS},
		{Metric: runrecord.ResourcePeakHostBytes, Value: peakHostBytes},
	}
	measures = append(measures, extraMeasures...)
	return runrecord.NewResourceFitness(runrecord.ResourceFitness{
		Scope: runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation, Model: model, Hardware: plan.body.Environment,
			Workload: plan.identity, Attempt: attempt,
		},
		Measures: measures,
		Interactions: &runrecord.InteractionWork{
			Allocations: allocations, WallNS: wallNS, ResourcePeakBytes: peakHostBytes,
		},
	})
}

func accumulateTranscriptionResources(summary *TranscriptionResourceSummary, measurement TranscriptionResourceMeasurement) {
	summary.Runs++
	summary.AudioSeconds += float64(measurement.SampleCount) / float64(measurement.SampleRate)
	summary.WallSeconds += float64(measurement.WallNS) / float64(time.Second)
	summary.Allocations += measurement.Allocations
	summary.PeakHostBytes = max(summary.PeakHostBytes, measurement.PeakHostBytes)
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
		{Name: "allocations", Value: float64(report.Summary.Allocations), Unit: "count", Direction: runrecord.DirectionMinimize},
		{Name: "audio-duration", Value: report.Summary.AudioSeconds, Unit: "seconds", Direction: runrecord.DirectionNeutral},
		{Name: "cold-load-allocations", Value: float64(report.Load.Allocations), Unit: "count", Direction: runrecord.DirectionMinimize},
		{Name: "cold-load-peak-host-bytes", Value: float64(report.Load.PeakHostBytes), Unit: "bytes", Direction: runrecord.DirectionMinimize},
		{Name: "cold-load-wall", Value: float64(report.Load.WallNS) / float64(time.Second), Unit: "seconds", Direction: runrecord.DirectionMinimize},
		{Name: "peak-host-bytes", Value: float64(report.Summary.PeakHostBytes), Unit: "bytes", Direction: runrecord.DirectionMinimize},
		{Name: "real-time-factor", Value: report.Summary.RealTimeFactor, Unit: "ratio", Direction: runrecord.DirectionMinimize},
		{Name: "timed-runs", Value: float64(report.Summary.Runs), Unit: "count", Direction: runrecord.DirectionNeutral},
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
		parents = append(parents, execution.Attempt, execution.Observation)
		if execution.Output.Valid() {
			parents = append(parents, execution.Output)
		}
	}
	parents = uniqueArtifactIDs(parents)
	alias := "evaluation/transcription-resource/report/" + report.Plan.String()
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

func readTranscriptionMemoryBoundary() transcriptionMemoryBoundary {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return transcriptionMemoryBoundary{heapAlloc: stats.HeapAlloc, mallocs: stats.Mallocs}
}

func elapsedResourceNanoseconds(started time.Time) uint64 {
	elapsed := time.Since(started).Nanoseconds()
	if elapsed <= 0 {
		return uint64(time.Nanosecond)
	}
	return uint64(elapsed)
}
