package densecausal

import (
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

func derivedTestLearningRate(t *testing.T, model *Model) float64 {
	t.Helper()
	plan := testcheck.Must(model.TrainingPlan())(t)
	// The fixtures train from their random initialization.
	return trainingprogram.BuiltinOptimizerPolicy().InitializationLearningRate(plan.ParameterCount())
}
