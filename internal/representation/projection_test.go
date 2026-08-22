package representation

import (
	"reflect"
	"testing"
)

func TestProjectPaddedF32BroadcastsProjectedZeroRow(t *testing.T) {
	projection, err := CompileTwoLayerProjectionF32(1, []float32{2}, []float32{0}, []float32{1}, []float32{3})
	if err != nil {
		t.Fatal(err)
	}
	got, err := ProjectPaddedF32([]float32{1}, 1, 3, projection, false)
	if err != nil {
		t.Fatal(err)
	}
	if got[1] != got[2] || reflect.DeepEqual(got, []float32{0, 0, 0}) {
		t.Fatalf("projection = %v", got)
	}
}

func TestCompileTwoLayerProjectionF32RejectsGeometryMismatch(t *testing.T) {
	if _, err := CompileTwoLayerProjectionF32(2, []float32{1, 2}, []float32{1, 2}, []float32{1}, []float32{1}); err == nil {
		t.Fatal("expected projection geometry error")
	}
}
