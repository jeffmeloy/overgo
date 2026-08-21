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
	return trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(plan.ParameterCount())
}
