package inference

import (
	"slices"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

func TestLayerInputCapturePreservesRequestedOrder(t *testing.T) {
	capture := testcheck.Must(newLayerInputCapture([]int32{2, 0}, 3))(t)
	capture.set(0, reference.Value{Shape: tensor.MustShape(2, 2), Data: []float32{1, 2, 3, 4}})
	capture.set(2, reference.Value{Shape: tensor.MustShape(2, 2), Data: []float32{5, 6, 7, 8}})
	result := testcheck.Must(capture.result(2, 2))(t)
	want := []float32{5, 6, 1, 2, 7, 8, 3, 4}
	if !result.Shape.Equal(tensor.MustShape(4, 2)) || !slices.Equal(result.Data, want) {
		t.Fatalf("captured layer inputs = %v/%v, want %v", result.Shape.Slice(), result.Data, want)
	}
}

func TestLayerInputCaptureRejectsMissingLayer(t *testing.T) {
	capture := testcheck.Must(newLayerInputCapture([]int32{1}, 2))(t)
	if _, err := capture.result(2, 1); err == nil {
		t.Fatal("accepted missing captured layer")
	}
	if _, err := newLayerInputCapture([]int32{2}, 2); err == nil {
		t.Fatal("accepted out-of-range capture layer")
	}
}
