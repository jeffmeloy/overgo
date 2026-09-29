package runrecord

import "slices"

// CanonicalOutcomes is the closed execution-outcome vocabulary in
// declaration order; ValidOutcome is membership in exactly this set.
var CanonicalOutcomes = []Outcome{
	OutcomeSucceeded, OutcomeFailed, OutcomeCancelled, outcomeRefused,
	outcomeSkipped, outcomeInapplicable, outcomeSuperseded,
	OutcomeInconclusive, outcomeLost, OutcomeRecovered,
}

// validOutcome reports membership in the canonical vocabulary.
func validOutcome(outcome Outcome) bool {
	return slices.Contains(CanonicalOutcomes, outcome)
}
