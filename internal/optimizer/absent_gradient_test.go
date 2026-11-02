package optimizer

import (
	"maps"
	"slices"
	"testing"
)

// absentGeometry is the shape of each of the fixture's tensors.
func absentGeometry(string, int) (int, int, error) { return 2, 2, nil }

// stepAbsentGradient trains two tensors one step on both and one on one of them through step, gathering as a trainer does,
// and holds the tensor with no gradient where the first step left it, on whichever backend step takes.
func stepAbsentGradient(t *testing.T, tensors map[string][]float32, pack *TensorPack, step func()) {
	t.Helper()
	gradient := []float32{1, 2, 3, 4}
	if err := pack.GatherGradients(map[string][]float32{"absent": gradient, "present": gradient}); err != nil {
		t.Fatal(err)
	}
	step()
	pack.Scatter()
	stepped := maps.Clone(tensors)
	for name, values := range tensors {
		stepped[name] = slices.Clone(values)
	}
	if err := pack.GatherGradients(map[string][]float32{"present": gradient}); err != nil {
		t.Fatal(err)
	}
	step()
	pack.Scatter()
	if !slices.Equal(tensors["absent"], stepped["absent"]) {
		t.Errorf("the tensor with no gradient moved from %v to %v", stepped["absent"], tensors["absent"])
	}
	if slices.Equal(tensors["present"], stepped["present"]) {
		t.Errorf("the tensor with a gradient stayed at %v", tensors["present"])
	}
	// A gather that holds it again steps it.
	if err := pack.GatherGradients(map[string][]float32{"absent": gradient, "present": gradient}); err != nil {
		t.Fatal(err)
	}
	before := slices.Clone(tensors["absent"])
	step()
	pack.Scatter()
	if slices.Equal(tensors["absent"], before) {
		t.Errorf("the tensor that took a gradient again stayed at %v", before)
	}
}

// TestAbsentGradientKeepsItsTensorOnTheHost holds a pack's host optimizer to the gathers it reads.
func TestAbsentGradientKeepsItsTensorOnTheHost(t *testing.T) {
	t.Parallel()
	tensors := map[string][]float32{"absent": {1, 1, 1, 1}, "present": {1, 1, 1, 1}}
	pack, err := NewTensorPack(tensors, absentGeometry)
	if err != nil {
		t.Fatal(err)
	}
	optimizer, err := pack.NewOptimizer(Config{BaseLearningRate: 0.1, Momentum: 0.9, Schedule: ScheduleConstant})
	if err != nil {
		t.Fatal(err)
	}
	stepAbsentGradient(t, tensors, pack, func() { optimizer.Step() })
}
