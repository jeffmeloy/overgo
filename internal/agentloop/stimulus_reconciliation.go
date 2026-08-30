package agentloop

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

// ReconcileLateStimuli admits at most one follow-up run for a consumed
// boundary. Repeated notifications — reconnect flaps, duplicate deliveries —
// coalesce into one pending reconciliation per boundary: a caller whose
// signal joins an inflight run resolves the durable admission instead of
// re-admitting, and a signal landing mid-run stays visibly pending for the
// next reconciliation, so no admitted wakeup is lost.
func (c *Coordinator) ReconcileLateStimuli(ctx context.Context, boundary artifact.ID, sources []artifact.ID) (runrecord.StimulusFollowup, bool, error) {
	key := boundary.String()
	c.wakeups.Signal(key)
	var followup runrecord.StimulusFollowup
	var admitted bool
	ran, err := c.wakeups.Reconcile(ctx, key, func(ctx context.Context) error {
		reconciled, fresh, reconcileErr := workflowruntime.ReconcileLateStimuli(ctx, c.store, boundary, sources)
		followup, admitted = reconciled, fresh
		return reconcileErr
	})
	if err != nil {
		return runrecord.StimulusFollowup{}, false, err
	}
	if ran {
		return followup, admitted, nil
	}
	current, _, err := (runrecord.StimulusFollowupAuthority{Repository: c.store}).Current(ctx, boundary)
	return current, false, err
}
