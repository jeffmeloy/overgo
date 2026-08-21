// Package trainingsession supervises training runs as model-sessions: when a
// run carries an observation store, the whole run executes under the
// capability-runtime director — a component-session lease is the resource
// admission (a refused lease means the run never touches weights), phase
// walls and process-level device samples accumulate through the run, and the
// result is one typed ServingObservation with the training task, committed
// with its environment so the training claim can carry session-measured
// provenance. The package also bootstraps the token-training recipe authority
// such a run resolves.
package trainingworkflow

import (
	"context"
	"fmt"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// Observer: one supervised training run under a director lease.
type ProbeObserver struct {
	store       artifact.Repository
	environment runrecord.Environment
	director    *capabilityruntime.ModelSessionDirector[struct{}, struct{}, struct{}]
	lease       *capabilityruntime.SessionLease[struct{}]
	sampler     *sessionSampler
	started     time.Time
	phases      []runrecord.PhaseMetric
	hardware    []runrecord.ServingHardwareSample
	// stepUsed/stepElapsed: the largest device-global residency seen at any
	// step boundary, recorded as the run's single mid-run hardware sample.
	stepUsed    uint64
	stepElapsed uint64
}

// New builds the environment identity and the training session director;
// admission happens once the model identity is known.
func NewProbeObserver(store artifact.Repository, host bool) (*ProbeObserver, error) {
	device, backend := "cuda0", "cuda-resident"
	if host {
		device, backend = "cpu", "host"
	}
	environment, err := runrecord.CurrentEnvironment(device, backend)
	if err != nil {
		return nil, fmt.Errorf("training session: environment: %w", err)
	}
	director, err := capabilityruntime.NewComponentSessionDirector[struct{}]("training", device, 1)
	if err != nil {
		return nil, fmt.Errorf("training session: director: %w", err)
	}
	observer := &ProbeObserver{store: store, environment: environment, director: director, started: time.Now()}
	if !host {
		observer.sampler = newSessionSampler()
	}
	return observer, nil
}

// SessionSnapshot returns the training admission and component lifecycle state.
func (o *ProbeObserver) SessionSnapshot() capabilityruntime.SessionSnapshot {
	if o == nil || o.director == nil {
		return capabilityruntime.SessionSnapshot{}
	}
	return o.director.Snapshot()
}

// Admit leases the exclusive training component session -- admission before
// residency: a refused lease means the run never loads weights.
func (o *ProbeObserver) Admit(ctx context.Context, model, recipeID artifact.ID) error {
	if o == nil {
		return nil
	}
	lease, err := o.director.LeaseComponent(ctx, modelrecipe.ComponentSession{
		Identity: recipeID, Node: "training", Module: "training.session",
		Model: model, Session: recipe.SessionRequest,
	}, func(context.Context) (struct{}, error) { return struct{}{}, nil })
	if err != nil {
		return fmt.Errorf("training session: admission refused: %w", err)
	}
	o.lease = lease
	o.sampleHardware(runrecord.ServingHardwareStart)
	return nil
}

// Phase records one lifecycle stage wall.
func (o *ProbeObserver) Phase(phase runrecord.Phase, wall time.Duration) {
	if o == nil || wall <= 0 {
		return
	}
	o.phases = append(o.phases, runrecord.PhaseMetric{Phase: phase, DurationNS: uint64(wall)})
}

// sampleHardware appends a device-global residency sample for one lifecycle
// stage.
func (o *ProbeObserver) sampleHardware(stage runrecord.ServingHardwareStage) {
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

// SampleStep records device-global residency at a step boundary; the largest
// reading becomes the run's single mid-run hardware sample.
func (o *ProbeObserver) SampleStep() {
	if o == nil {
		return
	}
	used, ok := o.sampler.sample()
	if !ok || used <= o.stepUsed {
		return
	}
	o.stepUsed, o.stepElapsed = used, uint64(time.Since(o.started))
}

// Finish releases the lease, closes the director, and commits the typed
// observation with its environment. The observation identity returns for
// the run's evidence.
func (o *ProbeObserver) Finish(
	ctx context.Context, model, recipeID artifact.ID, runErr error, streamPosition uint64,
) (artifact.ID, error) {
	if o == nil {
		return artifact.ID{}, nil
	}
	if o.stepUsed > 0 {
		o.hardware = append(o.hardware, runrecord.ServingHardwareSample{
			Stage: runrecord.ServingHardwarePrefill, ElapsedNS: o.stepElapsed,
			DeviceCurrentBytes: o.stepUsed, DevicePeakBytes: o.stepUsed,
		})
	}
	o.sampleHardware(runrecord.ServingHardwareFinish)
	o.sampler.close()
	if err := o.lease.Release(); err != nil {
		return artifact.ID{}, fmt.Errorf("training session: release: %w", err)
	}
	if err := o.director.Close(ctx); err != nil {
		return artifact.ID{}, fmt.Errorf("training session: close: %w", err)
	}
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
		return artifact.ID{}, fmt.Errorf("training session: commit environment: %w", err)
	}
	published, err := runrecord.PublishServingObservation(ctx, o.store, observation)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("training session: publish observation: %w", err)
	}
	return published.ID, nil
}
