package hostmath

import (
	"math"
	"slices"
	"testing"
)

func TestSelectedLogProbIntoMatchesMaterialized(t *testing.T) {
	const rows, input, output = 2, 3, 4
	x := []float32{0.5, -1, 2, 1.5, 0.25, -0.5}
	weight := []float32{1, 0, -1, 0.5, 1, 0, -0.5, 0.25, 1, 1, -1, 0.5}
	bias := []float32{0.25, -0.5, 0.75, 0}
	targets := []int{2, 0}
	selected := []bool{true, true}
	got := make([]float64, rows)
	if err := SelectedLogProbInto(got, x, weight, bias, targets, selected, rows, input, output, make([]float32, input)); err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, rows*output)
	Linear(logits, x, weight, rows, input, output)
	for row := range rows {
		AddBias(logits[row*output:(row+1)*output], bias)
		SoftmaxInPlace(logits[row*output : (row+1)*output])
		want := math.Log(float64(logits[row*output+targets[row]]))
		if math.Abs(got[row]-want) > 1e-6 {
			t.Fatalf("row %d log probability=%g want=%g", row, got[row], want)
		}
	}
}

func TestSelectedLogProbBackwardFiniteDifference(t *testing.T) {
	const rows, input, output = 1, 2, 3
	x := []float32{0.75, -0.25}
	weight := []float32{0.5, 1, -1, 0.25, 0.75, -0.5}
	targets := []int{1}
	selected := []bool{true}
	dX := make([]float32, len(x))
	dWeight := make([]float32, len(weight))
	if err := SelectedLogProbBackward(dX, dWeight, nil, x, weight, nil, targets, selected, []float64{1}, rows, input, output, make([]float32, output), false); err != nil {
		t.Fatal(err)
	}
	dXOnly := make([]float32, len(x))
	if err := SelectedLogProbBackward(dXOnly, nil, nil, x, weight, nil, targets, selected, []float64{1}, rows, input, output, make([]float32, output), false); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(dXOnly, dX) {
		t.Fatalf("activation-only gradient=%v want=%v", dXOnly, dX)
	}
	const epsilon = float32(1e-3)
	for index := range weight {
		original := weight[index]
		weight[index] = original + epsilon
		plus := make([]float64, rows)
		_ = SelectedLogProbInto(plus, x, weight, nil, targets, selected, rows, input, output, make([]float32, output))
		weight[index] = original - epsilon
		minus := make([]float64, rows)
		_ = SelectedLogProbInto(minus, x, weight, nil, targets, selected, rows, input, output, make([]float32, output))
		weight[index] = original
		want := (plus[0] - minus[0]) / (2 * float64(epsilon))
		if math.Abs(float64(dWeight[index])-want) > 2e-4 {
			t.Fatalf("gradient %d=%g want=%g", index, dWeight[index], want)
		}
	}
}

func TestSelectedLogProbRejectsInvalidContract(t *testing.T) {
	if err := SelectedLogProbInto(make([]float64, 1), []float32{1}, []float32{1}, nil, []int{1}, []bool{true}, 1, 1, 1, []float32{0}); err == nil {
		t.Fatal("accepted target beyond projection")
	}
	if err := SelectedLogProbInto(make([]float64, 1), []float32{1}, []float32{1}, nil, []int{0}, []bool{true}, 1, 1, 1, nil); err == nil {
		t.Fatal("accepted absent workspace")
	}
}
