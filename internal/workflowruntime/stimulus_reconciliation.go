package workflowruntime

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// ReconcileLateStimuli converts duplicate operational wakeups into one durable
// follow-up admission owned by the consumed attempt boundary.
func ReconcileLateStimuli(ctx context.Context, repository artifact.Repository, boundary artifact.ID, sources []artifact.ID) (runrecord.StimulusFollowup, bool, error) {
	return (runrecord.StimulusFollowupAuthority{Repository: repository}).Admit(ctx, boundary, sources)
}
