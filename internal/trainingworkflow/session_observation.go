package trainingworkflow

import (
	"context"
	"fmt"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// sessionObserver: one Execute run under a director lease.
type sessionObserver struct {
	store       artifact.Repository
	environment runrecord.Environment
	sampler     *sessionSampler
	started     time.Time
	phases      []runrecord.PhaseMetric
	hardware    []runrecord.ServingHardwareSample
	// stepUsed/stepElapsed: the largest device-global residency seen at any
	// step boundary, recorded as the run's single mid-run hardware sample.
	stepUsed    uint64
	stepElapsed uint64
	stepSampled bool
}

func newSessionObserver(store artifact.Repository, host bool) (*sessionObserver, error) {
	device, backend := "cuda0", "cuda-resident"
	if host {
		device, backend = "cpu", "host"
	}
	environment, err := runrecord.CurrentEnvironment(device, backend)
	if err != nil {
		return nil, fmt.Errorf("training workflow: session environment: %w", err)
	}
	observer := &sessionObserver{store: store, environment: environment, started: time.Now()}
	if !host {
		observer.sampler = newSessionSampler()
	}
	return observer, nil
}

func (o *sessionObserver) phase(phase runrecord.Phase, wall time.Duration) {
	if o == nil {
		return
	}
	o.phases = append(o.phases, runrecord.PhaseMetric{Phase: phase, DurationNS: uint64(wall)})
}

func (o *sessionObserver) sampleHardware(stage runrecord.ServingHardwareStage) {
	if o == nil {
		return
	}
	used, ok := o.sampler.sample()
	if !ok {
		return
	}
	o.hardware = append(o.hardware, runrecord.ServingHardwareSample{
		Stage: stage, ElapsedNS: uint64(time.Since(o.started)),
		DeviceCurrentBytes: used, DevicePeakBytes: used,
	})
}

func (o *sessionObserver) sampleStep() {
	if o == nil {
		return
	}
	used, ok := o.sampler.sample()
	if !ok || used <= o.stepUsed {
		return
	}
	o.stepUsed, o.stepElapsed, o.stepSampled = used, uint64(time.Since(o.started)), true
}

func (o *sessionObserver) finish(
	ctx context.Context, model, recipeID artifact.ID, runErr error, streamPosition uint64,
) (artifact.ID, error) {
	if o == nil {
		return artifact.ID{}, nil
	}
	if o.stepSampled {
		o.hardware = append(o.hardware, runrecord.ServingHardwareSample{
			Stage: runrecord.ServingHardwarePrefill, ElapsedNS: o.stepElapsed,
			DeviceCurrentBytes: o.stepUsed, DevicePeakBytes: o.stepUsed,
		})
	}
	o.sampleHardware(runrecord.ServingHardwareFinish)
	o.sampler.close()
	outcome, failure := runrecord.OutcomeSucceeded, ""
	if runErr != nil {
		outcome, failure = runrecord.OutcomeFailed, "training_failed"
	}
	var peakDevice uint64
	for _, sample := range o.hardware {
		peakDevice = max(peakDevice, sample.DevicePeakBytes)
	}
	observation := runrecord.ServingObservation{
		Model: model, Recipe: recipeID, Environment: o.environment.ID,
		Task: recipe.TaskTraining, Outcome: outcome, Failure: failure,
		StartedUnixNS: o.started.UnixNano(), MeasuredNS: uint64(time.Since(o.started)),
		Usage:     runrecord.ServingUsage{InputTokens: streamPosition},
		Resources: runrecord.ServingResources{PeakDeviceBytes: peakDevice},
		Phases:    o.phases, Hardware: o.hardware,
	}
	environmentContent, err := o.environment.Content()
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := o.store.Commit(ctx, artifact.Batch{
		Key: "training-session-environment/" + o.environment.ID.String(), Contents: []artifact.Content{environmentContent},
	}); err != nil {
		return artifact.ID{}, fmt.Errorf("training workflow: commit session environment: %w", err)
	}
	published, err := runrecord.PublishServingObservation(ctx, o.store, observation)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("training workflow: publish session observation: %w", err)
	}
	return published.ID, nil
}
