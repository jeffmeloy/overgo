package trainingworkflow

import (
	"context"
	"fmt"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// Observer: one measured training session.
type Observer struct {
	store       artifact.Repository
	environment runrecord.Environment
	director    *capabilityruntime.ModelSessionDirector[struct{}, struct{}, struct{}]
	lease       *capabilityruntime.SessionLease[struct{}]
	sampler     *sessionSampler
	started     time.Time
	phases      []runrecord.PhaseMetric
	hardware    []runrecord.ServingHardwareSample
	stepUsed    uint64
	stepElapsed uint64
	stepSampled bool
	// pre holds the evidence standing when the session was admitted, so
	// Finish can publish the bracket: improvement as the delta between
	// committed records, never a narrated claim.
	pre        BracketSlice
	bracketing bool
}

// NewObserver starts session measurement.
func NewObserver(store artifact.Repository, host bool) (*Observer, error) {
	device, backend := "cuda0", "cuda-resident"
	if host {
		device, backend = "cpu", "host"
	}
	environment, err := runrecord.CurrentEnvironment(device, backend)
	if err != nil {
		return nil, fmt.Errorf("training observation: environment: %w", err)
	}
	observer := &Observer{store: store, environment: environment, started: time.Now()}
	if !host {
		observer.sampler = newSessionSampler()
	}
	return observer, nil
}

// Admit acquires directed training ownership.
func (o *Observer) Admit(ctx context.Context, model, recipeID artifact.ID) error {
	if o == nil {
		return nil
	}
	director, err := capabilityruntime.NewComponentSessionDirector[struct{}](
		"training", o.environment.Device, recipe.SessionRequestCapacity,
	)
	if err != nil {
		return fmt.Errorf("training observation: director: %w", err)
	}
	lease, err := director.LeaseComponent(ctx, modelrecipe.ComponentSession{
		Identity: recipeID, Node: "training", Module: "training.session",
		Model: model, Session: recipe.SessionRequest,
	}, func(context.Context) (struct{}, error) { return struct{}{}, nil })
	if err != nil {
		_ = director.Close(ctx)
		return fmt.Errorf("training observation: admission refused: %w", err)
	}
	o.director, o.lease = director, lease
	// Capture the evidence standing before training so the session's
	// bracket can state its deltas against committed records.
	if store, typed := o.store.(*overgodb.Store); typed {
		o.pre, o.bracketing = captureBracketSlice(ctx, store, model, recipeID), true
	}
	o.sampleHardware(runrecord.ServingHardwareStart)
	return nil
}

// Phase records one lifecycle wall.
func (o *Observer) Phase(phase runrecord.Phase, wall time.Duration) {
	if o == nil || wall <= 0 {
		return
	}
	o.phases = append(o.phases, runrecord.PhaseMetric{Phase: phase, DurationNS: uint64(wall)})
}

func (o *Observer) sampleHardware(stage runrecord.ServingHardwareStage) {
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

// SampleStep retains peak sampled residency.
func (o *Observer) SampleStep() {
	if o == nil {
		return
	}
	used, ok := o.sampler.sample()
	if !ok || o.stepSampled && used <= o.stepUsed {
		return
	}
	o.stepUsed, o.stepElapsed, o.stepSampled = used, uint64(time.Since(o.started)), true
}

// Finish releases directed ownership; publishes evidence.
func (o *Observer) Finish(
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
	if o.lease != nil {
		if err := o.lease.Release(); err != nil {
			return artifact.ID{}, fmt.Errorf("training observation: release: %w", err)
		}
	}
	if o.director != nil {
		if err := o.director.Close(ctx); err != nil {
			return artifact.ID{}, fmt.Errorf("training observation: close: %w", err)
		}
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
	// An identical environment from an earlier session is already the
	// stored fact; re-running the same configuration must observe, not
	// refuse on the no-op environment batch.
	recorded := false
	if presence, answers := o.store.(artifact.ContentPresence); answers {
		present, err := presence.PresentContents(ctx, []artifact.ID{o.environment.ID})
		if err != nil {
			return artifact.ID{}, fmt.Errorf("training observation: environment presence: %w", err)
		}
		recorded = len(present) == 1
	}
	if !recorded {
		if _, err := o.store.Commit(ctx, artifact.Batch{
			Key:      "training-session-environment/" + o.environment.ID.String(),
			Contents: []artifact.Content{environmentContent},
		}); err != nil {
			return artifact.ID{}, fmt.Errorf("training observation: commit environment: %w", err)
		}
	}
	published, err := runrecord.PublishServingObservation(ctx, o.store, observation)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("training observation: publish: %w", err)
	}
	// The bracket binds this session to the evidence before and after
	// it. A bracket that cannot publish must not fail the session whose
	// observation already committed -- it reports and stands aside.
	if store, typed := o.store.(*overgodb.Store); typed && o.bracketing {
		post := captureBracketSlice(ctx, store, model, recipeID)
		if _, err := publishTrainingBracket(ctx, o.store, published.ID, model, recipeID, o.pre, post); err != nil {
			fmt.Printf("training observation: bracket: %v\n", err)
		}
	}
	return published.ID, nil
}
