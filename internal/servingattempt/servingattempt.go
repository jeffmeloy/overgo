// Package servingattempt records one serving attempt as the observation the
// user scorecard counts, and parks a failed one as a finding. The server,
// its generation workspace and the recipe command's run and measure verbs
// record through it, so a page's attempt and a command line's count alike.
// Recording is local: it writes only to the store the server or command was
// opened on, and a server without a store records nothing.
package servingattempt

import (
	"context"
	"errors"
	"fmt"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/finding"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// OutputEnvironment names an invocation-owned file that recipe measure writes
// with its exact published serving observation identity, also on failure.
const OutputEnvironment = "OVERGO_SERVING_ATTEMPT_OUTPUT"

// WholeOutput is one execution of a capability that returns its
// output whole: the model and recipe it ran, the run it recorded if any,
// when it started, how long its output took, how long of that its model
// took to load when the execution loaded one, the device memory it peaked
// at and the error it ended with. PeakMeasured says the peak was measured,
// so that a zero is an execution that held no device memory, on the host,
// rather than one nothing measured. An execution whose model decoded its
// output token by token, a text returned whole all the same, also has how
// long of the wall that decode took and the tokens it gave; any other has
// neither.
type WholeOutput struct {
	Model, Recipe, Run artifact.ID
	Task               recipe.Task
	Started            time.Time
	Wall, Load, Decode time.Duration
	OutputTokens       uint64
	PeakDeviceBytes    uint64
	PeakMeasured       bool
	Cause              error
}

// failureCode is the code a failed attempt records.
const failureCode = "generation_failed"

// RecordWholeOutput publishes the attempt as the serving observation
// the scorecard counts, under the environment it ran in: a cancel as
// cancelled, any other error as a failure. The published observation is returned
// even when parking its failure as a finding fails.
func RecordWholeOutput(ctx context.Context, store artifact.Repository, attempt WholeOutput) (runrecord.ServingObservation, error) {
	environment, err := runrecord.CurrentEnvironment("process", "go")
	if err != nil {
		return runrecord.ServingObservation{}, err
	}
	batch, err := environment.Batch("serving/environment/" + environment.ID.String())
	if err != nil {
		return runrecord.ServingObservation{}, err
	}
	// The environment of an earlier attempt on this machine is already the
	// stored fact.
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return runrecord.ServingObservation{}, err
	}
	observation := runrecord.ServingObservation{
		Version: artifact.InitialDocumentVersion, Model: attempt.Model, Recipe: attempt.Recipe, Environment: environment.ID,
		Run: attempt.Run, Task: attempt.Task, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: attempt.Started.UnixNano(),
		MeasuredNS: uint64(max(attempt.Wall, 0)), Resources: runrecord.ServingResources{PeakDeviceBytes: attempt.PeakDeviceBytes},
		Usage: runrecord.ServingUsage{OutputTokens: attempt.OutputTokens},
	}
	if attempt.Load > 0 {
		observation.Phases = []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: uint64(attempt.Load)}}
	}
	if attempt.Decode > 0 {
		observation.Phases = append(observation.Phases, runrecord.PhaseMetric{Phase: runrecord.PhaseDecode, DurationNS: uint64(attempt.Decode)})
	}
	if attempt.PeakMeasured && attempt.PeakDeviceBytes == 0 {
		// The resources' peak cannot tell a measured zero from an absent
		// one; a sample of the device at the finish, holding nothing, can.
		observation.Hardware = []runrecord.ServingHardwareSample{{Stage: runrecord.ServingHardwareFinish, ElapsedNS: observation.MeasuredNS}}
	}
	switch {
	case errors.Is(attempt.Cause, context.Canceled) || errors.Is(attempt.Cause, context.DeadlineExceeded):
		observation.Outcome = runrecord.OutcomeCancelled
	case attempt.Cause != nil:
		observation.Outcome, observation.Failure = runrecord.OutcomeFailed, failureCode
	}
	return Record(ctx, store, observation)
}

// Record publishes one serving observation, the attempt the scorecard's
// failure rate counts, and parks a failed one as an open finding that cites
// it and its recipe: a failure a user met becomes the loop's work beside the
// scorecard's gaps. It returns the published observation even when the
// finding could not be parked. An attempt whose run is bound to an
// environment is recorded under that one, where the work was done, and not
// the recorder's own, where it was only watched from; an attempt with no
// run, a run that cannot be read or one that names none keeps the recorder's.
func Record(ctx context.Context, store artifact.Repository, observation runrecord.ServingObservation) (runrecord.ServingObservation, error) {
	if run, err := runrecord.RequireRun(ctx, store, observation.Run); err == nil && run.Environment.Valid() {
		observation.Environment = run.Environment
	}
	published, err := runrecord.PublishServingObservation(ctx, store, observation)
	if err != nil || published.Outcome != runrecord.OutcomeFailed {
		return published, err
	}
	_, batch, err := finding.NewTextBatch(
		fmt.Sprintf("A %s serving attempt failed: %s", published.Task, published.Failure),
		finding.SeverityHigh, []string{published.Recipe.String()}, []string{published.ID.String()},
		"fix what the failed attempt records; the recipe's failure-rate cell and the loop's failing-attempts row close with it",
		fmt.Sprintf("go run ./cmd/validate -scorecard shows a failed %s attempt under the recipe", published.Task),
	)
	if err == nil {
		_, err = artifact.Publish(ctx, store, batch)
	}
	return published, err
}
