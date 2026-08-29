package trainingworkflow

import (
	"context"
	"fmt"
	"math"
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
	id, _, _, err := o.finish(ctx, model, recipeID, runErr, streamPosition, Result{})
	return id, err
}

func (o *Observer) finishTraining(
	ctx context.Context,
	model, recipeID artifact.ID,
	runErr error,
	result Result,
) (artifact.ID, []artifact.ID, artifact.ID, error) {
	return o.finish(ctx, model, recipeID, runErr, result.StreamPosition, result)
}

func (o *Observer) finish(
	ctx context.Context,
	model, recipeID artifact.ID,
	runErr error,
	streamPosition uint64,
	result Result,
) (artifact.ID, []artifact.ID, artifact.ID, error) {
	if o == nil {
		return artifact.ID{}, nil, artifact.ID{}, nil
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
			return artifact.ID{}, nil, artifact.ID{}, fmt.Errorf("training observation: release: %w", err)
		}
	}
	if o.director != nil {
		if err := o.director.Close(ctx); err != nil {
			return artifact.ID{}, nil, artifact.ID{}, fmt.Errorf("training observation: close: %w", err)
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
	measured := uint64(time.Since(o.started))
	observation := runrecord.ServingObservation{
		Model: model, Recipe: recipeID, Environment: o.environment.ID,
		Task: recipe.TaskTraining, Outcome: outcome, Failure: failure,
		StartedUnixNS: o.started.UnixNano(), MeasuredNS: measured,
		Usage:     runrecord.ServingUsage{InputTokens: streamPosition},
		Resources: runrecord.ServingResources{PeakDeviceBytes: peakDevice},
		Phases:    o.phases, Hardware: o.hardware,
	}
	environmentContent, err := o.environment.Content()
	if err != nil {
		return artifact.ID{}, nil, artifact.ID{}, err
	}
	if _, err := o.store.Commit(ctx, artifact.Batch{
		Key:      "training-session-environment/" + o.environment.ID.String(),
		Contents: []artifact.Content{environmentContent},
	}); err != nil {
		return artifact.ID{}, nil, artifact.ID{}, fmt.Errorf("training observation: commit environment: %w", err)
	}
	var published runrecord.ServingObservation
	var routerIDs []artifact.ID
	var routerCoverage artifact.ID
	if len(result.router) == 0 {
		published, err = runrecord.PublishServingObservation(ctx, o.store, observation)
	} else {
		published, routerIDs, routerCoverage, err = o.publishRouterTrainingEvidence(ctx, observation, result, measured)
	}
	if err != nil {
		return artifact.ID{}, nil, artifact.ID{}, fmt.Errorf("training observation: publish: %w", err)
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
	return published.ID, routerIDs, routerCoverage, nil
}

func (o *Observer) publishRouterTrainingEvidence(
	ctx context.Context,
	serving runrecord.ServingObservation,
	result Result,
	measured uint64,
) (runrecord.ServingObservation, []artifact.ID, artifact.ID, error) {
	if result.Checkpoint.ID().Kind() != artifact.KindCheckpoint {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, fmt.Errorf("router evidence requires a committed checkpoint")
	}
	codeCommit, err := executableCodeCommit(".")
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	code, err := runrecord.NewCodeRevision(codeCommit)
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	run, err := runrecord.NewBoundRun(
		serving.Recipe, runrecord.OutcomeSucceeded,
		[]artifact.ID{
			result.authority.model, result.authority.data.Dataset, result.authority.data.Split,
			result.authority.program, result.authority.runPlan,
		},
		[]artifact.ID{result.Checkpoint.ID()}, "", code.Commit, serving.Environment,
		measured, serving.Phases,
	)
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	serving.Run = run.ID
	published, err := runrecord.NewServingObservation(serving)
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	codeBatch, err := code.Publication()
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	runBatch, err := run.Batch("training/run/" + run.ID.String())
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	servingBatch, err := published.Batch("training/observation/" + published.ID.String())
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	batch := artifact.Batch{Key: "training/router-observations/" + run.ID.String()}
	// Checkpoint and compiled recipes are exact external publications. Register
	// their identities so this atomic evidence graph resolves without copying
	// checkpoint bytes into OvergoDB.
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: result.Checkpoint.ID()},
		artifact.Descriptor{ID: result.authority.program},
		artifact.Descriptor{ID: result.authority.runPlan},
	)
	appendBatch := func(part artifact.Batch) {
		batch.Artifacts = append(batch.Artifacts, part.Artifacts...)
		batch.Contents = append(batch.Contents, part.Contents...)
		batch.Manifests = append(batch.Manifests, part.Manifests...)
		batch.Lineage = append(batch.Lineage, part.Lineage...)
		batch.Causality = append(batch.Causality, part.Causality...)
		batch.Aliases = append(batch.Aliases, part.Aliases...)
		batch.Locations = append(batch.Locations, part.Locations...)
	}
	appendBatch(codeBatch)
	appendBatch(runBatch)
	appendBatch(servingBatch)
	routerIDs := make([]artifact.ID, 0, len(result.router))
	routerValues := make([]runrecord.MoERouterObservation, 0, len(result.router))
	for _, record := range result.router {
		raw, err := decodeRouterObservation(record)
		if err != nil {
			return runrecord.ServingObservation{}, nil, artifact.ID{}, err
		}
		value, err := trainingRouterObservation(raw, run.ID, serving.Recipe, code.ID, result)
		if err != nil {
			return runrecord.ServingObservation{}, nil, artifact.ID{}, err
		}
		content, err := value.Content()
		if err != nil {
			return runrecord.ServingObservation{}, nil, artifact.ID{}, err
		}
		part, err := runrecord.RouterObservationBatch(value, content)
		if err != nil {
			return runrecord.ServingObservation{}, nil, artifact.ID{}, err
		}
		appendBatch(part)
		routerValues = append(routerValues, value)
		routerIDs = append(routerIDs, value.ID)
	}
	coverage, err := runrecord.NewMoERouterObservationCoverage(
		routerValues, result.routerExpectation.firstStep, result.routerExpectation.steps, result.routerExpectation.layers,
	)
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	coverageBatch, err := coverage.Batch(ctx, o.store)
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	appendBatch(coverageBatch)
	if _, err := artifact.CommitBatch(ctx, o.store, batch); err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	return published, routerIDs, coverage.ID, nil
}

var executableCodeCommit = runrecord.ExecutableCodeCommit

func trainingRouterObservation(
	raw routerStepObservation,
	run, recipeID, code artifact.ID,
	result Result,
) (runrecord.MoERouterObservation, error) {
	if raw.Step < 0 || raw.Observation.Layer < 0 || raw.Observation.Experts < 0 || raw.Observation.TopK < 0 ||
		uint64(raw.Observation.Layer) > math.MaxUint32 || uint64(raw.Observation.Experts) > math.MaxUint32 ||
		uint64(raw.Observation.TopK) > math.MaxUint32 {
		return runrecord.MoERouterObservation{}, fmt.Errorf("router evidence has invalid geometry")
	}
	selections := make([]uint32, len(raw.Observation.Selections))
	for index, expert := range raw.Observation.Selections {
		if expert < 0 || uint64(expert) > math.MaxUint32 {
			return runrecord.MoERouterObservation{}, fmt.Errorf("router evidence has invalid expert")
		}
		selections[index] = uint32(expert)
	}
	margins := make([]runrecord.MoERouterMargin, len(raw.Observation.Margins))
	for index, margin := range raw.Observation.Margins {
		margins[index] = runrecord.MoERouterMargin{Observed: margin.Observed, Value: margin.Value}
	}
	return runrecord.NewMoERouterObservation(runrecord.MoERouterObservation{
		Run: run, Model: result.authority.model,
		Dataset: result.authority.data.Dataset, Split: result.authority.data.Split,
		Recipe: recipeID, Code: code, Checkpoint: result.Checkpoint.ID(), Policy: result.authority.runPlan,
		Step: uint64(raw.Step), Layer: uint32(raw.Observation.Layer),
		Experts: uint32(raw.Observation.Experts), TopK: uint32(raw.Observation.TopK),
		Selections: selections, CombineWeights: raw.Observation.CombineWeights,
		Accepted: raw.Observation.Accepted, Margins: margins,
	})
}
