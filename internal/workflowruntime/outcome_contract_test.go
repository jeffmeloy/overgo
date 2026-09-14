package workflowruntime

import (
	"testing"

	"overgo/internal/runrecord"
	"overgo/internal/testvocab"
)

// TestCanonicalExecutionOutcomeContract refuses a local fork of the
// canonical execution-outcome vocabulary: this package expresses how
// executions end with runrecord.Outcome, never a restated string.
func TestCanonicalExecutionOutcomeContract(t *testing.T) {
	testvocab.ForbidVocabularyRedefinition(t, ".", runrecord.CanonicalOutcomes)
}
