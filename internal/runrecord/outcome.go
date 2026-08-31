package runrecord

import "slices"

// CanonicalOutcomes is the closed execution-outcome vocabulary in
// declaration order; ValidOutcome is membership in exactly this set.
var CanonicalOutcomes = []Outcome{
	OutcomeSucceeded, OutcomeFailed, OutcomeCancelled, OutcomeRefused,
	OutcomeSkipped, OutcomeInapplicable, OutcomeSuperseded,
	OutcomeInconclusive, OutcomeLost, OutcomeRecovered,
}

// ValidOutcome reports membership in the canonical vocabulary.
func ValidOutcome(outcome Outcome) bool {
	return slices.Contains(CanonicalOutcomes, outcome)
}
