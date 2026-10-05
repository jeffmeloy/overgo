package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/driver"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

// TrainingEvidencePublication is the single workflow-owned atomic publication
// request for training evidence assembled through shared run-record contracts.
type TrainingEvidencePublication struct {
	Batch artifact.Batch
}

// PublishTrainingEvidence admits exactly one semantic batch. Producers may
// prepare typed facts and raw chunks, but cannot create another commit path.
func PublishTrainingEvidence(ctx context.Context, repository artifact.Repository, publication TrainingEvidencePublication) error {
	if ctx == nil || repository == nil || !strings.HasPrefix(publication.Batch.Key, "training/evidence/") {
		return errors.New("training workflow: training evidence publication is incomplete")
	}
	_, err := artifact.Publish(ctx, repository, publication.Batch)
	return err
}

// Observer: one measured training session.
type Observer struct {
	store       artifact.Repository
	environment runrecord.Environment
	director    *capabilityruntime.ModelSessionDirector[struct{}, struct{}, struct{}]
	lease       *capabilityruntime.SessionLease[struct{}]
	started     time.Time
	clock       processmeasure.Stopwatch
	walls       processmeasure.Walls
	phases      []runrecord.PhaseMetric
	hardware    []runrecord.ServingHardwareSample
	stepUsed    uint64
	stepElapsed uint64
	stepSampled bool
	// host marks a session that runs on the host processor: it holds no device
	// memory, whatever else its process does. process reads the device memory
	// the process holds and the most it has held, which a device session reads
	// as its own.
	host    bool
	process func() (current, peak uint64)
	// pre holds the evidence standing when the session was admitted, so
	// Finish can publish the bracket: improvement as the delta between
	// committed records, never a narrated claim.
	pre        BracketSlice
	bracketing bool
	// finished marks a session whose outcome is recorded; a later Finish
	// records nothing and releases nothing again.
	finished bool
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
	return &Observer{store: store, environment: environment, started: time.Now(), clock: processmeasure.NewStopwatch(), host: host, process: processLedger}, nil
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

// processLedger reads the driver's process-wide ledger: the device memory this
// process holds and the most it has held.
func processLedger() (current, peak uint64) {
	return driver.ProcessCurrentBytes(), driver.ProcessPeakBytes()
}

// held is the device memory the session holds and the most it has held: the
// process ledger for a device session, which a training process alone owns, and
// zero for a host one, which holds none whatever else its process does (a
// workbench trains on the host beside a resident inference runner). Zero is a
// measurement.
func (o *Observer) held() (current, peak uint64) {
	if o.host {
		return 0, 0
	}
	return o.process()
}

// sampleHardware records the device memory the session holds and the most it
// has held, never the device's use beside other processes.
func (o *Observer) sampleHardware(stage runrecord.ServingHardwareStage) {
	if o == nil {
		return
	}
	current, peak := o.held()
	o.hardware = append(o.hardware, runrecord.ServingHardwareSample{
		Stage: stage, ElapsedNS: o.walls.Elapsed(o.clock), DeviceCurrentBytes: current, DevicePeakBytes: peak,
	})
}

// SampleStep retains the step at which the process held the most.
func (o *Observer) SampleStep() {
	if o == nil {
		return
	}
	used, _ := o.held()
	if o.stepSampled && used <= o.stepUsed {
		return
	}
	o.stepUsed, o.stepElapsed, o.stepSampled = used, o.walls.Elapsed(o.clock), true
}

// Finish releases directed ownership and publishes evidence, once: a
// session finished already returns no observation.
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
	if o == nil || o.finished {
		return artifact.ID{}, nil, artifact.ID{}, nil
	}
	o.finished = true
	if o.stepSampled {
		o.hardware = append(o.hardware, runrecord.ServingHardwareSample{
			Stage: runrecord.ServingHardwarePrefill, ElapsedNS: o.stepElapsed,
			DeviceCurrentBytes: o.stepUsed, DevicePeakBytes: o.stepUsed,
		})
	}
	o.sampleHardware(runrecord.ServingHardwareFinish)
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
		if errors.Is(runErr, context.Canceled) {
			outcome, failure = runrecord.OutcomeCancelled, ""
		}
	}
	var peakDevice uint64
	for _, sample := range o.hardware {
		peakDevice = max(peakDevice, sample.DevicePeakBytes)
	}
	measured := o.walls.Elapsed(o.clock)
	if err := o.walls.Err; err != nil {
		return artifact.ID{}, nil, artifact.ID{}, err
	}
	inputTokens := streamPosition
	if result.Objective == trainingprogram.ObjectiveCTC {
		// A speech stream cursor counts source records, not tokenizer units.
		// Keep unmeasured token usage absent from the shared observation.
		inputTokens = 0
	}
	observation := runrecord.ServingObservation{
		Model: model, Recipe: recipeID, Environment: o.environment.ID,
		Task: recipe.TaskTraining, Outcome: outcome, Failure: failure,
		StartedUnixNS: o.started.UnixNano(), MeasuredNS: measured,
		Usage:     runrecord.ServingUsage{InputTokens: inputTokens},
		Resources: runrecord.ServingResources{PeakDeviceBytes: peakDevice},
		Phases:    o.phases, Hardware: o.hardware,
	}
	environmentContent, err := o.environment.Content()
	if err != nil {
		return artifact.ID{}, nil, artifact.ID{}, err
	}
	// An identical environment from an earlier session is already the
	// stored fact; re-running the same configuration must observe, not
	// refuse on the no-op environment batch.
	recorded := false
	if presence, answers := o.store.(artifact.ContentPresence); answers {
		present, err := presence.PresentContents(ctx, []artifact.ID{o.environment.ID})
		if err != nil {
			return artifact.ID{}, nil, artifact.ID{}, fmt.Errorf("training observation: environment presence: %w", err)
		}
		recorded = len(present) == 1
	}
	if !recorded {
		if _, err := o.store.Commit(ctx, artifact.Batch{
			Key:      "training-session-environment/" + o.environment.ID.String(),
			Contents: []artifact.Content{environmentContent},
		}); err != nil {
			return artifact.ID{}, nil, artifact.ID{}, fmt.Errorf("training observation: commit environment: %w", err)
		}
	}
	var published runrecord.ServingObservation
	var routerIDs []artifact.ID
	var routerCoverage artifact.ID
	var post BracketSlice
	if store, typed := o.store.(*overgodb.Store); typed && o.bracketing {
		post = captureBracketSlice(ctx, store, model, recipeID)
	}
	if len(result.router) == 0 {
		published, err = runrecord.PublishServingObservation(ctx, o.store, observation)
	} else {
		published, routerIDs, routerCoverage, err = o.publishRouterTrainingEvidence(
			ctx, observation, result, measured, o.pre, post, o.bracketing,
		)
	}
	if err != nil {
		return artifact.ID{}, nil, artifact.ID{}, fmt.Errorf("training observation: publish: %w", err)
	}
	// The bracket binds this session to the evidence before and after
	// it. A bracket that cannot publish must not fail the session whose
	// observation already committed -- it reports and stands aside.
	if _, typed := o.store.(*overgodb.Store); typed && o.bracketing && len(result.router) == 0 {
		if _, err := publishTrainingBracket(ctx, o.store, published.ID, model, recipeID, o.pre, post); err != nil {
			fmt.Printf("training observation: bracket: %v\n", err)
		}
	}
	return published.ID, routerIDs, routerCoverage, nil
}

// trainingVerificationReason is the activation decision a training run's
// verification records.
const trainingVerificationReason = "a training run under the recipe succeeded and published what it trained"

// VerificationOutputEnvironment names an invocation-owned file for the exact
// completed gate/run pair. Normal training leaves it unset; acquisition reads
// it only after the producer has completed, never from rendered messages.
const VerificationOutputEnvironment = "OVERGO_TRAIN_VERIFICATION"

// Verify records the finished session's success under recipeID as the
// recipe's verification and activates the recipe on it, or reverifies the
// active recipe: a gate bound to the code revision the run executed, the
// environment the session observed and the session's wall, whose run
// depends on evidence, the session's observation and what the run
// published. A run whose code revision cannot be named, as from a modified
// tree, leaves the recipe's lifecycle as it was and says so.
func (o *Observer) Verify(ctx context.Context, recipeID artifact.ID, evidence ...artifact.ID) error {
	if o == nil {
		return nil
	}
	definition, err := recipe.RequireDefinition(ctx, o.store, recipeID)
	if err != nil {
		return err
	}
	revision, err := executableCodeCommit(".")
	if err != nil {
		fmt.Printf("training recipe %s not verified: %v\n", recipeID, err)
		return nil
	}
	wall := max(uint64(time.Since(o.started)), 1)
	record, err := runrecord.NewGateRecord(recipeID, o.environment.ID, revision, runrecord.OutcomeSucceeded, "", wall, []runrecord.GateStep{{
		Name: "training-run", Phase: runrecord.PhaseForwardBackward, Outcome: runrecord.StepSucceeded, DurationNS: wall,
	}})
	if err != nil {
		return err
	}
	run := record.Run
	if record.Run, err = runrecord.NewBoundRun(run.Recipe, run.Outcome, evidence, run.Outputs, run.Failure, run.CodeCommit, run.Environment, run.MeasuredNS, run.Phases); err != nil {
		return err
	}
	batch, err := record.Batch("training/evidence/verification/" + recipeID.String() + "/" + record.Run.ID.String())
	if err != nil {
		return err
	}
	if err := PublishTrainingEvidence(ctx, o.store, TrainingEvidencePublication{Batch: batch}); err != nil {
		return err
	}
	verification := modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}
	if err := modelrecipe.ActivateCapability(ctx, o.store, definition, verification, recipe.EvidenceVerified, trainingVerificationReason); err != nil {
		return err
	}
	if path := os.Getenv(VerificationOutputEnvironment); path != "" {
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, clioptions.PrivateFileMode)
		if err != nil {
			return fmt.Errorf("training verification handoff: %w", err)
		}
		if err := errors.Join(clioptions.WriteJSON(file, verification), file.Close()); err != nil {
			return fmt.Errorf("training verification handoff: %w", err)
		}
	}
	fmt.Printf("training recipe verified and active: %s (run %s at %.12s)\n", recipeID, record.Run.ID, revision)
	return nil
}

func (o *Observer) publishRouterTrainingEvidence(
	ctx context.Context,
	serving runrecord.ServingObservation,
	result Result,
	measured uint64,
	pre, post BracketSlice,
	bracket bool,
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
	batch := artifact.Batch{Key: "training/evidence/" + run.ID.String()}
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
		routerValues = append(routerValues, value)
	}
	chunk, err := runrecord.NewMoERouterObservationChunk(routerValues, artifact.ID{})
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	coverage, err := runrecord.NewMoERouterObservationCoverage(
		chunk, result.routerExpectation.firstStep, result.routerExpectation.steps, result.routerExpectation.layers,
	)
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	coverageBatch, err := coverage.Batch(ctx, o.store, chunk)
	if err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	appendBatch(coverageBatch)
	if bracket {
		_, bracketBatch, err := trainingBracketBatch(published.ID, result.authority.model, serving.Recipe, pre, post)
		if err != nil {
			return runrecord.ServingObservation{}, nil, artifact.ID{}, err
		}
		appendBatch(bracketBatch)
	}
	if err := PublishTrainingEvidence(ctx, o.store, TrainingEvidencePublication{Batch: batch}); err != nil {
		return runrecord.ServingObservation{}, nil, artifact.ID{}, err
	}
	return published, []artifact.ID{chunk.ID}, coverage.ID, nil
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
