package tensor

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

func TestCompatible(t *testing.T) {
	left := &Tensor{Type: dtype.F32, Shape: MustShape(2, 3)}
	right := &Tensor{Type: dtype.F32, Shape: MustShape(2, 3)}
	if !Compatible(left, right) {
		t.Fatal("equal tensors are incompatible")
	}
	right.Shape = MustShape(3, 2)
	if Compatible(left, right) {
		t.Fatal("different shapes are compatible")
	}
}
