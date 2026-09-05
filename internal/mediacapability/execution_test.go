package mediacapability

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

// candidateCapabilityExecution compiles the candidate execution a test
// runs a capability against, the way verify does before activation.
func candidateCapabilityExecution(
	t testing.TB,
	store artifact.Reader,
	program recipe.Program,
) modelrecipe.CapabilityEvidenceSelection {
	t.Helper()
	execution, err := modelrecipe.CompileCandidateExecution(t.Context(), store, program)
	if err != nil {
		t.Fatal(err)
	}
	return execution
}
