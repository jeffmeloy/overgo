package densecausal

import (
	"testing"

	"overgo/internal/trainingprogram"
)

func derivedTestLearningRate(t *testing.T, model *Model) float64 {
	t.Helper()
	plan, err := model.TrainingPlan()
	if err != nil {
		t.Fatal(err)
	}
	// The fixtures train from their random initialization.
	return trainingprogram.BuiltinOptimizerPolicy().InitializationLearningRate(plan.ParameterCount())
}
