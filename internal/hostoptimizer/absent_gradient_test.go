package hostoptimizer

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestAbsentGradientLeavesAGroupAlone: a group told to have received no gradient keeps its weights and its momentum, where
// its normalized momentum would take a full-sized step on nothing, and the step count moves for the rest.
func TestAbsentGradientLeavesAGroupAlone(t *testing.T) {
	t.Parallel()
	const elements, side = 8, 2
	weights, gradients := make([]float32, elements), make([]float32, elements)
	plan, err := CompilePlan(elements, []GroupSpec{
		{Name: "absent", Start: 0, End: elements / 2, Rows: side, Cols: side},
		{Name: "present", Start: elements / 2, End: elements, Rows: side, Cols: side},
	})
	testcheck.NoError(t, err)
	optimizer, err := New(weights, gradients, plan, Config{BaseLearningRate: 0.1, Momentum: 0.9, Schedule: ScheduleConstant})
	testcheck.NoError(t, err)
	fill := func() {
		for index := range gradients {
			gradients[index] = float32(index + 1)
		}
	}
	fill()
	optimizer.Step()
	stepped, momentum := slices.Clone(weights), slices.Clone(optimizer.Momentum())
	optimizer.Receiving(func(group Group) bool { return group.Name != "absent" })
	fill()
	clear(gradients[:elements/2])
	result := optimizer.Step()
	if !slices.Equal(weights[:elements/2], stepped[:elements/2]) || !slices.Equal(optimizer.Momentum()[:elements/2], momentum[:elements/2]) {
		t.Fatalf("the absent group moved: weights %v from %v, momentum %v from %v", weights[:elements/2], stepped[:elements/2], optimizer.Momentum()[:elements/2], momentum[:elements/2])
	}
	if slices.Equal(weights[elements/2:], stepped[elements/2:]) || result.Step != 2 {
		t.Fatalf("the present group did not step: weights %v, step %d", weights[elements/2:], result.Step)
	}
}
