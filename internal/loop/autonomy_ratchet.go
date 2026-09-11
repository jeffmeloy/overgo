//overgo:runtime-inputs caller

package loop

import (
	"errors"
	"fmt"
)

const (
	// AutonomyBudgetFloor is the bounded start and the rollback target: one
	// attempt, the same conservative bound an empty measured history
	// derives, because a regressing or unmeasured driver has no evidence
	// supporting more.
	AutonomyBudgetFloor = 1
	// AutonomyRatchetStep is the canary policy's single ordinal step: each
	// positive review widens the budget by exactly one attempt, so autonomy
	// grows one measured notch at a time and never jumps.
	AutonomyRatchetStep = 1
)

// AutonomyRatchetDecision records one review of the driver's autonomy:
// the budget before and after, whether it widened or rolled back, and the
// exact condition outcomes behind the decision.
type AutonomyRatchetDecision struct {
	CurrentBudget uint64   `json:"current_budget"`
	NextBudget    uint64   `json:"next_budget"`
	Widened       bool     `json:"widened"`
	RolledBack    bool     `json:"rolled_back"`
	Reasons       []string `json:"reasons"`
}

// RatchetDriverAutonomy widens the driver budget only on positive,
// calibrated, safe evidence, reusing the three recorded instruments
// rather than a second safety owner: the learning curve must show
// promotions with positive fitness per compute and a late hit rate at
// least the early one, the enumeration ranker's calibration must hold,
// and live safety must be quiet — no provided breaker state above
// advisory. All conditions passing widens the budget by exactly one
// attempt; any condition failing rolls the budget back to the
// conservative floor, with every condition's outcome recorded either way.
func RatchetDriverAutonomy(
	currentBudget uint64,
	curve DriverLearningCurve,
	calibration CandidateCalibration,
	safety []CircuitBreakerTransition,
) (AutonomyRatchetDecision, error) {
	if currentBudget < AutonomyBudgetFloor {
		return AutonomyRatchetDecision{}, errors.New("loop: the autonomy ratchet requires the current bounded budget")
	}
	if curve.Attempts == 0 {
		return AutonomyRatchetDecision{}, errors.New("loop: the autonomy ratchet requires a derived learning curve")
	}
	decision := AutonomyRatchetDecision{CurrentBudget: currentBudget}
	positive := true
	if curve.Promotions == 0 || curve.FitnessPerCompute <= 0 {
		positive = false
		decision.Reasons = append(decision.Reasons, "learning curve carries no positive measured fitness per compute")
	} else if curve.LateHitRate < curve.EarlyHitRate {
		positive = false
		decision.Reasons = append(decision.Reasons, fmt.Sprintf(
			"hit rate fell from %.3f to %.3f across the record", curve.EarlyHitRate, curve.LateHitRate,
		))
	} else {
		decision.Reasons = append(decision.Reasons, "learning curve positive")
	}
	if !calibration.Trustworthy {
		positive = false
		decision.Reasons = append(decision.Reasons, "enumeration ranker calibration does not hold")
	} else {
		decision.Reasons = append(decision.Reasons, "calibration holds")
	}
	quiet := true
	for _, transition := range safety {
		if transition.Level != BreakerClear && transition.Level != BreakerAdvisory {
			quiet = false
			decision.Reasons = append(decision.Reasons, fmt.Sprintf(
				"live safety is not quiet: metric %q stands at %q", transition.Metric, transition.Level,
			))
		}
	}
	if quiet {
		decision.Reasons = append(decision.Reasons, "live safety quiet")
	}
	if positive && quiet {
		decision.NextBudget = currentBudget + AutonomyRatchetStep
		decision.Widened = true
		return decision, nil
	}
	decision.NextBudget = AutonomyBudgetFloor
	decision.RolledBack = currentBudget > AutonomyBudgetFloor
	return decision, nil
}
