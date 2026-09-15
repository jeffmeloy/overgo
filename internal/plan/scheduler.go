package plan

import (
	"fmt"
	"sort"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/worklease"
)

// LeaseRecommendation is one abandoned-lease reconciliation verdict:
// recommendation-only by contract -- the scheduler advises retries and
// reports measurement error, it never cancels, restarts or mutates anything.
type LeaseRecommendation struct {
	Lease            artifact.ID `json:"lease"`
	Task             string      `json:"task"`
	Experiment       artifact.ID `json:"experiment,omitzero"`
	Abandoned        bool        `json:"abandoned"`
	RecommendedRetry uint32      `json:"recommended_retry,omitzero"`
	ResumeCheckpoint artifact.ID `json:"resume_checkpoint,omitzero"`
	RecoveryNS       uint64      `json:"recovery_ns,omitzero"`
	WallErrorNS      int64       `json:"wall_error_ns,omitzero"`
	Reason           string      `json:"reason"`
}

// ReconcileExperimentLeases derives recommendation-only reconciliation from
// committed leases and measured outcomes. A lease past its heartbeat expiry
// with no measured outcome is abandoned: the recommendation names the retry
// identity a recovery must use and the checkpoint it resumes from, and
// carries the measured recovery cost when an outcome recorded one. Measured
// leases report prediction error so estimates improve from evidence.
func ReconcileExperimentLeases(
	leases []worklease.Lease,
	outcomes []LeaseOutcome,
	now time.Time,
) []LeaseRecommendation {
	measured := make(map[artifact.ID]LeaseOutcome, len(outcomes))
	for _, outcome := range outcomes {
		measured[outcome.Lease] = outcome
	}
	recommendations := make([]LeaseRecommendation, 0, len(leases))
	for _, lease := range leases {
		recommendation := LeaseRecommendation{
			Lease: lease.ID, Task: lease.Task, Experiment: lease.Experiment,
		}
		outcome, hasOutcome := measured[lease.ID]
		expires, err := time.Parse(time.RFC3339Nano, lease.ExpiresAt)
		expired := err == nil && now.After(expires)
		switch {
		case hasOutcome && outcome.Abandoned:
			recommendation.Abandoned = true
			recommendation.RecommendedRetry = lease.Retry + 1
			recommendation.ResumeCheckpoint = lease.Checkpoint
			recommendation.RecoveryNS = outcome.RecoveryNS
			recommendation.Reason = "outcome recorded abandonment; retry from the lease checkpoint with the incremented identity"
		case hasOutcome:
			if lease.PredictedWallNS > 0 {
				recommendation.WallErrorNS = int64(outcome.ActualWallNS) - int64(lease.PredictedWallNS)
			}
			recommendation.Reason = fmt.Sprintf(
				"measured: wall %s vs predicted %s",
				time.Duration(outcome.ActualWallNS), time.Duration(lease.PredictedWallNS))
		case expired:
			recommendation.Abandoned = true
			recommendation.RecommendedRetry = lease.Retry + 1
			recommendation.ResumeCheckpoint = lease.Checkpoint
			recommendation.Reason = "heartbeat expired with no measured outcome; treat as abandoned and retry from the lease checkpoint"
		default:
			recommendation.Reason = "live within its heartbeat bound"
		}
		recommendations = append(recommendations, recommendation)
	}
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].Task < recommendations[j].Task
	})
	return recommendations
}
