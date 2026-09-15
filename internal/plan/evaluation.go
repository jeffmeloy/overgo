package plan

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/worklease"
)

// SchedulingEvaluation compares deterministic resource predictions with both
// measured use and the owner's approved lease. Bounds are maxima observed in
// this finite sample, not confidence intervals.
type SchedulingEvaluation struct {
	Samples                  int                 `json:"samples"`
	RecommendationAgreements int                 `json:"recommendation_agreements"`
	RecommendationWins       int                 `json:"recommendation_wins"`
	OwnerWins                int                 `json:"owner_wins"`
	Ties                     int                 `json:"ties"`
	Reserved                 worklease.Resources `json:"reserved"`
	Actual                   worklease.Resources `json:"actual"`
	MaxResourceError         worklease.Resources `json:"max_resource_error"`
	MaxWallErrorNS           uint64              `json:"max_wall_error_ns"`
	MaxInterferenceErrorNS   uint64              `json:"max_interference_error_ns"`
	Collisions               int                 `json:"collisions"`
	Abandonments             int                 `json:"abandonments"`
	TotalRecoveryNS          uint64              `json:"total_recovery_ns"`
}

// EvaluateScheduling evaluates evidence only; it never creates or assigns a
// lease. A recommendation wins only when it is no worse in every resource
// dimension and strictly closer in at least one.
func EvaluateScheduling(outcomes []LeaseOutcome, leases map[artifact.ID]worklease.Lease) (SchedulingEvaluation, error) {
	result := SchedulingEvaluation{Samples: len(outcomes)}
	for _, outcome := range outcomes {
		lease, ok := leases[outcome.Lease]
		if !ok {
			return SchedulingEvaluation{}, errors.New("plan: scheduling evaluation lacks referenced lease")
		}
		if outcome.Predicted == lease.Resources {
			result.RecommendationAgreements++
		}
		addResources(&result.Reserved, lease.Resources)
		addResources(&result.Actual, outcome.Actual)
		predictedError := resourceError(outcome.Predicted, outcome.Actual)
		ownerError := resourceError(lease.Resources, outcome.Actual)
		maxResources(&result.MaxResourceError, predictedError)
		switch compareErrors(predictedError, ownerError) {
		case -1:
			result.RecommendationWins++
		case 1:
			result.OwnerWins++
		default:
			result.Ties++
		}
		result.MaxWallErrorNS = max(result.MaxWallErrorNS, distance(outcome.PredictedWallNS, outcome.ActualWallNS))
		result.MaxInterferenceErrorNS = max(result.MaxInterferenceErrorNS, distance(outcome.PredictedInterferenceNS, outcome.ActualInterferenceNS))
		if outcome.Collision {
			result.Collisions++
		}
		if outcome.Abandoned {
			result.Abandonments++
		}
		result.TotalRecoveryNS += outcome.RecoveryNS
	}
	return result, nil
}

func resourceError(left, right worklease.Resources) worklease.Resources {
	return worklease.Resources{
		CPUThreads: abs(left.CPUThreads - right.CPUThreads),
		HostRAMGiB: abs(left.HostRAMGiB - right.HostRAMGiB),
		VRAMGiB:    abs(left.VRAMGiB - right.VRAMGiB),
	}
}

func compareErrors(recommendation, owner worklease.Resources) int {
	recommendationBetter := recommendation.CPUThreads <= owner.CPUThreads && recommendation.HostRAMGiB <= owner.HostRAMGiB && recommendation.VRAMGiB <= owner.VRAMGiB
	ownerBetter := owner.CPUThreads <= recommendation.CPUThreads && owner.HostRAMGiB <= recommendation.HostRAMGiB && owner.VRAMGiB <= recommendation.VRAMGiB
	switch {
	case recommendationBetter && recommendation != owner:
		return -1
	case ownerBetter && recommendation != owner:
		return 1
	default:
		return 0
	}
}

func addResources(total *worklease.Resources, value worklease.Resources) {
	total.CPUThreads += value.CPUThreads
	total.HostRAMGiB += value.HostRAMGiB
	total.VRAMGiB += value.VRAMGiB
}

func maxResources(bound *worklease.Resources, value worklease.Resources) {
	bound.CPUThreads = max(bound.CPUThreads, value.CPUThreads)
	bound.HostRAMGiB = max(bound.HostRAMGiB, value.HostRAMGiB)
	bound.VRAMGiB = max(bound.VRAMGiB, value.VRAMGiB)
}

func distance(left, right uint64) uint64 {
	if left > right {
		return left - right
	}
	return right - left
}

func abs(value int) int {
	if value < 0 {
		return -value
	}
	return value
}
