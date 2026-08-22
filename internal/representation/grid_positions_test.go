package representation

import (
	"reflect"
	"testing"
)

func TestAxisGridPositions(t *testing.T) {
	got := AxisGridPositions(1, 2, 2)
	if !reflect.DeepEqual(got[0], []uint32{0, 0, 0, 0, 0}) ||
		!reflect.DeepEqual(got[1], []uint32{0, 0, 0, 1, 1}) ||
		!reflect.DeepEqual(got[2], []uint32{0, 0, 1, 0, 1}) {
		t.Fatalf("positions = %v", got)
	}
}
