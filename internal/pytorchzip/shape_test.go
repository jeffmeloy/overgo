package pytorchzip

import (
	"slices"
	"testing"
)

func TestHostShape(t *testing.T) {
	shape, err := HostShape(TensorMeta{Name: "weight", Shape: []int64{2, 3}}, 2)
	if err != nil || !slices.Equal(shape, []int{2, 3}) {
		t.Fatalf("shape=%v err=%v", shape, err)
	}
	if _, err := HostShape(TensorMeta{Name: "weight", Shape: []int64{2, 0}}, 2); err == nil {
		t.Fatal("accepted zero extent")
	}
}
