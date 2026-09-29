// Package servingattempt records one execution of a capability that returns
// its output whole, an image, a clip, a prediction, as the serving attempt
// the user scorecard counts. The server's generation workspace and the
// recipe command's run and measure verbs record through it, so a page's
// generation and a command line's count alike.
package servingattempt

import (
	"context"
	"errors"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// WholeOutput is one execution of a capability that returns its
// output whole: the model and recipe it ran, the run it recorded if any,
// when it started, how long its output took, the device memory it peaked
// at and the error it ended with.
type WholeOutput struct {
	Model, Recipe, Run artifact.ID
	Task               recipe.Task
	Started            time.Time
	Wall               time.Duration
	PeakDeviceBytes    uint64
	Cause              error
}

// failureCode is the code a failed attempt records.
const failureCode = "generation_failed"

// RecordWholeOutput publishes the attempt as the serving observation
// the scorecard counts, under the environment it ran in: a cancel as
// cancelled, any other error as a failure.
func RecordWholeOutput(ctx context.Context, store artifact.Repository, attempt WholeOutput) error {
	environment, err := runrecord.CurrentEnvironment("process", "go")
	if err != nil {
		return err
	}
	batch, err := environment.Batch("serving/environment/" + environment.ID.String())
	if err != nil {
		return err
	}
	// The environment of an earlier attempt on this machine is already the
	// stored fact.
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return err
	}
	observation := runrecord.ServingObservation{
		Version: artifact.InitialDocumentVersion, Model: attempt.Model, Recipe: attempt.Recipe, Environment: environment.ID,
		Run: attempt.Run, Task: attempt.Task, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: attempt.Started.UnixNano(),
		MeasuredNS: uint64(max(attempt.Wall, 0)), Resources: runrecord.ServingResources{PeakDeviceBytes: attempt.PeakDeviceBytes},
	}
	switch {
	case errors.Is(attempt.Cause, context.Canceled) || errors.Is(attempt.Cause, context.DeadlineExceeded):
		observation.Outcome = runrecord.OutcomeCancelled
	case attempt.Cause != nil:
		observation.Outcome, observation.Failure = runrecord.OutcomeFailed, failureCode
	}
	_, err = runrecord.PublishServingObservation(ctx, store, observation)
	return err
}
