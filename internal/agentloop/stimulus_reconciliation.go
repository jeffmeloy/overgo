package agentloop

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

// ReconcileLateStimuli admits at most one follow-up run for a consumed
// boundary; repeated notifications resolve the existing durable admission.
func (c *Coordinator) ReconcileLateStimuli(ctx context.Context, boundary artifact.ID, sources []artifact.ID) (runrecord.StimulusFollowup, bool, error) {
	return workflowruntime.ReconcileLateStimuli(ctx, c.store, boundary, sources)
}
