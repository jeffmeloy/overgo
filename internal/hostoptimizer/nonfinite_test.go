package hostoptimizer

import (
	"math"
	"testing"
)

// TestAStepReportsANonFiniteDirection: a gradient holding a NaN or an infinity moves no weight of its group and says so,
// where the Frobenius guard reads false for a NaN and the group and its momentum became NaN. Another group steps.
func TestAStepReportsANonFiniteDirection(t *testing.T) {
	t.Parallel()
	const elements, side = 8, 2
	for name, planted := range map[string]float32{"NaN": float32(math.NaN()), "infinity": float32(math.Inf(1))} {
		weights, gradients := make([]float32, elements), make([]float32, elements)
		for index := range gradients {
			weights[index], gradients[index] = 1, float32(index+1)
		}
		gradients[0] = planted
		plan, err := CompilePlan(elements, []GroupSpec{
			{Name: "broken", Start: 0, End: elements / 2, Rows: side, Cols: side},
			{Name: "finite", Start: elements / 2, End: elements, Rows: side, Cols: side},
		})
		if err != nil {
			t.Fatal(err)
		}
		optimizer, err := New(weights, gradients, plan, Config{BaseLearningRate: 0.1, Momentum: 0.9, Schedule: ScheduleConstant})
		if err != nil {
			t.Fatal(err)
		}
		if !optimizer.Step().Nonfinite {
			t.Errorf("%s: the step reports a finite direction", name)
		}
		for index, weight := range weights[:elements/2] {
			if weight != 1 {
				t.Errorf("%s: weight %d of the broken group became %g", name, index, weight)
			}
		}
		if weights[elements/2] == 1 {
			t.Errorf("%s: the finite group did not step", name)
		}
	}
	weights, gradients := []float32{1, 1, 1, 1}, []float32{1, 2, 3, 4}
	plan, err := CompilePlan(len(weights), []GroupSpec{{Name: "finite", Start: 0, End: len(weights), Rows: side, Cols: side}})
	if err != nil {
		t.Fatal(err)
	}
	optimizer, err := New(weights, gradients, plan, Config{BaseLearningRate: 0.1, Momentum: 0.9, Schedule: ScheduleConstant})
	if err != nil {
		t.Fatal(err)
	}
	if optimizer.Step().Nonfinite {
		t.Error("a finite step reports a non-finite direction")
	}
}
