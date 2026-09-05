package server

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/runrecord"
)

// recordFailedWorkflowRun records a workflow that did not complete as a
// run under keyPrefix, so the operation carries a durable receipt of what
// did not happen; a cancelled context records a cancellation, not a fault.
func recordFailedWorkflowRun(ctx context.Context, store artifact.Repository, keyPrefix string, recipeID artifact.ID, inputs []artifact.ID, failure string, cause error) (operation.Completion, error) {
	outcome := runrecord.OutcomeFailed
	if errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) {
		outcome, failure = runrecord.OutcomeCancelled, ""
	}
	run, err := runrecord.NewRun(recipeID, outcome, inputs, nil, failure)
	if err == nil {
		batch, batchErr := run.Batch(keyPrefix + run.ID.String())
		if batchErr == nil {
			_, batchErr = artifact.CommitBatch(context.WithoutCancel(ctx), store, batch)
		}
		err = errors.Join(err, batchErr)
	}
	return operation.Completion{Run: run.ID}, errors.Join(cause, err)
}
