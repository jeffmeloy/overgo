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
	_, err = Record(ctx, store, observation)
	return err
}

// Record publishes one serving observation, the attempt the scorecard's
// failure rate counts, and parks a failed one as an open finding that cites
// it and its recipe: a failure a user met becomes the loop's work beside the
// scorecard's gaps. It returns the published observation even when the
// finding could not be parked.
func Record(ctx context.Context, store artifact.Repository, observation runrecord.ServingObservation) (runrecord.ServingObservation, error) {
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
