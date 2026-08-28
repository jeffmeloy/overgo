package trainingworkflow

import (
	"testing"

	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestCanonicalExecutionOutcomeContract refuses a local fork of the
// canonical execution-outcome vocabulary: this package expresses how
// executions end with runrecord.Outcome, never a restated string.
func TestCanonicalExecutionOutcomeContract(t *testing.T) {
	testutil.ForbidVocabularyRedefinition(t, ".", runrecord.OutcomeVocabulary())
}
