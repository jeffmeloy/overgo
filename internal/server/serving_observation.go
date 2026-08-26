package server

import (
	"context"
	"errors"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

func openServingRepository(config Config) (*overgodb.Store, runrecord.Environment, error) {
	repository := config.Repository
	if repository == nil {
		return nil, runrecord.Environment{}, nil
	}
	var err error
	environment := config.Environment
	if !environment.ID.Valid() {
		environment, err = runrecord.CurrentEnvironment("process", "go")
	} else {
		err = environment.ValidateIdentity()
	}
	if err == nil {
		var batch artifact.Batch
		batch, err = environment.Batch("serving/environment/" + environment.ID.String())
		if err == nil {
			_, err = artifact.CommitBatch(context.Background(), repository, batch)
		}
	}
	return repository, environment, err
}

func (h *Handler) servingIdentity(task recipe.Task) (artifact.ID, artifact.ID, bool) {
	if h == nil || h.repository == nil || !h.environment.ID.Valid() {
		return artifact.ID{}, artifact.ID{}, false
	}
	inspector, ok := h.generator.(interface {
		RecipeRuntimeDescription(recipe.Task) (modelrecipe.RuntimeDescription, error)
	})
	if !ok {
		return artifact.ID{}, artifact.ID{}, false
	}
	description, err := inspector.RecipeRuntimeDescription(task)
	if err != nil || description.Identity.Model.Kind() != artifact.KindModel || description.Identity.Recipe.Kind() != artifact.KindRecipe {
		return artifact.ID{}, artifact.ID{}, false
	}
	return description.Identity.Model, description.Identity.Recipe, true
}

func (h *Handler) publishServing(ctx context.Context, observation runrecord.ServingObservation) artifact.ID {
	if h == nil || h.repository == nil {
		return artifact.ID{}
	}
	observation.Environment = h.environment.ID
	published, err := runrecord.PublishServingObservation(context.WithoutCancel(ctx), h.repository, observation)
	if err != nil {
		h.observationErrors.Add(1)
		return artifact.ID{}
	}
	return published.ID
}

func (h *Handler) executeObservedOperation(
	ctx context.Context,
	reporter operation.Reporter,
	task recipe.Task,
	recipeID artifact.ID,
	execute operation.Executor,
) (operation.Completion, error) {
	started := time.Now()
	completion, err := execute(ctx, reporter)
	modelID := h.modelArtifact
	if !modelID.Valid() {
		modelID, _, _ = h.servingIdentity(task)
	}
	if modelID.Kind() != artifact.KindModel || recipeID.Kind() != artifact.KindRecipe {
		return completion, err
	}
	outcome, failure := executionOutcome(err)
	if err == nil && completion.Run.Kind() != artifact.KindRun {
		outcome, failure = runrecord.OutcomeFailed, "operation_failed"
	}
	observation := runrecord.ServingObservation{
		Model: modelID, Recipe: recipeID, Operation: reporter.OperationID(),
		Task: task, Outcome: outcome, Failure: failure,
		StartedUnixNS: started.UnixNano(), MeasuredNS: uint64(max(time.Since(started).Nanoseconds(), 0)),
	}
	if completion.Run.Kind() == artifact.KindRun {
		observation.Run = completion.Run
	}
	reporter.Attempt(h.publishServing(ctx, observation))
	return completion, err
}

func executionOutcome(err error) (runrecord.Outcome, string) {
	switch {
	case err == nil:
		return runrecord.OutcomeSucceeded, ""
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return runrecord.OutcomeCancelled, ""
	default:
		return runrecord.OutcomeFailed, "execution_failed"
	}
}

func servingPhases(prompt, total time.Duration) []runrecord.PhaseMetric {
	var metrics []runrecord.PhaseMetric
	if prompt > 0 {
		metrics = append(metrics, runrecord.PhaseMetric{Phase: runrecord.PhasePrefill, DurationNS: uint64(prompt)})
	}
	if decode := total - prompt; decode > 0 {
		metrics = append(metrics, runrecord.PhaseMetric{Phase: runrecord.PhaseDecode, DurationNS: uint64(decode)})
	}
	return metrics
}

func servingTransferDelta(before, after uint64) uint64 {
	if after < before {
		return 0
	}
	return after - before
}

type servingHardwareCollector struct {
	started time.Time
	api     DeviceMemoryAPI
	samples []runrecord.ServingHardwareSample
	prefill bool
}

func newServingHardwareCollector(generator Generator, started time.Time) *servingHardwareCollector {
	api, _ := generator.(DeviceMemoryAPI)
	return &servingHardwareCollector{started: started, api: api}
}

func (collector *servingHardwareCollector) sample(ctx context.Context, stage runrecord.ServingHardwareStage) {
	if collector == nil || collector.api == nil || stage == runrecord.ServingHardwarePrefill && collector.prefill {
		return
	}
	stats, err := collector.api.DeviceMemoryStats(ctx)
	if err != nil {
		return
	}
	if stage == runrecord.ServingHardwarePrefill {
		collector.prefill = true
	}
	collector.samples = append(collector.samples, runrecord.ServingHardwareSample{
		Stage: stage, ElapsedNS: uint64(max(time.Since(collector.started).Nanoseconds(), 0)),
		DeviceCurrentBytes: stats.CurrentBytes, DevicePeakBytes: stats.PeakBytes,
		DeviceAllocations: stats.Allocations,
	})
}

func (collector *servingHardwareCollector) peakDeviceBytes() uint64 {
	var peak uint64
	for _, sample := range collector.samples {
		peak = max(peak, sample.DevicePeakBytes)
	}
	return peak
}
