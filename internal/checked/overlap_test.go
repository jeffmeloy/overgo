package checked

import "testing"

func TestSlicesOverlap(t *testing.T) {
	values := []float32{1, 2, 3, 4}
	for _, test := range []struct {
		left, right []float32
		want        bool
	}{
		{nil, values, false}, {values, nil, false}, {values[:2], values[2:], false},
		{values, values, true}, {values[:3], values[2:], true}, {values[2:], values[:3], true},
		{values, []float32{1, 2, 3, 4}, false},
	} {
		if got := SlicesOverlap(test.left, test.right); got != test.want {
			t.Fatalf("overlap(%v,%v)=%t want %t", test.left, test.right, got, test.want)
		}
	}
}
