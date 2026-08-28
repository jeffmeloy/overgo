package runrecord

import "slices"

// CanonicalOutcomes is the closed execution-outcome vocabulary in
// declaration order; ValidOutcome is membership in exactly this set.
var CanonicalOutcomes = []Outcome{
	OutcomeSucceeded, OutcomeFailed, OutcomeCancelled, OutcomeRefused,
	OutcomeSkipped, OutcomeInapplicable, OutcomeSuperseded,
	OutcomeInconclusive, OutcomeLost, OutcomeRecovered,
}

// OutcomeVocabulary returns the canonical members as strings for
// vocabulary guards.
func OutcomeVocabulary() []string {
	vocabulary := make([]string, len(CanonicalOutcomes))
	for index, outcome := range CanonicalOutcomes {
		vocabulary[index] = string(outcome)
	}
	return vocabulary
}

// ValidOutcome reports membership in the canonical vocabulary.
func ValidOutcome(outcome Outcome) bool {
	return slices.Contains(CanonicalOutcomes, outcome)
}

// StepOutcomeCanonical maps one gate step outcome onto the canonical
// vocabulary. Reuse is evidence provenance, not a distinct way to end:
// a reused step succeeded on exact cached evidence.
func StepOutcomeCanonical(step StepOutcome) (Outcome, bool) {
	switch step {
	case StepSucceeded, StepReused:
		return OutcomeSucceeded, true
	case StepFailed:
		return OutcomeFailed, true
	case StepCancelled:
		return OutcomeCancelled, true
	case StepSkipped:
		return OutcomeSkipped, true
	case StepInapplicable:
		return OutcomeInapplicable, true
	}
	return "", false
}

// LaneOutcomeCanonical maps one lane outcome onto the canonical
// vocabulary. Unavailable never passes: a lane that could not obtain
// its prerequisites failed its contract, by doctrine.
func LaneOutcomeCanonical(lane LaneOutcome) (Outcome, bool) {
	switch lane {
	case LanePassed:
		return OutcomeSucceeded, true
	case LaneFailed, LaneUnavailable:
		return OutcomeFailed, true
	case LaneEmpty:
		return OutcomeInapplicable, true
	}
	return "", false
}
