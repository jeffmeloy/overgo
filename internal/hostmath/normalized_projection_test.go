package hostmath

import (
	"reflect"
	"testing"
)

func TestNormalizeProjectRowsToChannelsF64(t *testing.T) {
	out := make([]float32, 4)
	err := NormalizeProjectRowsToChannelsF64(out, []float32{1, 2, 3, 4}, []float32{2, 1}, []float32{1, -1}, []float32{1, 1, -1, 2}, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out, []float32{4, 10, -1, -1}) {
		t.Fatalf("output = %v", out)
	}
}
