package media

import (
	"context"
	"math"
	"testing"
)

func TestResamplePolyReusesScratchAndRejectsOverlap(t *testing.T) {
	destination := make([]float32, 4)
	got, err := ResamplePoly(t.Context(), destination, []float32{1, 2}, 2, 1, []float64{0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 0, 2, 0}
	if &got[0] != &destination[0] {
		t.Fatal("scratch not reused")
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got=%v want=%v", got, want)
		}
	}
	if _, err := ResamplePoly(t.Context(), destination, destination, 1, 1, []float64{1}); err == nil {
		t.Fatal("accepted overlapping slices")
	}
	if _, err := ResamplePoly(t.Context(), destination[1:], destination[:2], 1, 1, []float64{1}); err == nil {
		t.Fatal("accepted offset overlap")
	}
}

func TestResamplePolyRejectsInvalidNumericsAndCancellation(t *testing.T) {
	for _, test := range []struct {
		input    []float32
		up, down int
		taps     []float64
	}{
		{[]float32{1}, 0, 1, []float64{1}},
		{[]float32{1}, 1, 1, []float64{1, 1}},
		{[]float32{float32(math.NaN())}, 1, 1, []float64{1}},
		{[]float32{1}, 1, 1, []float64{math.Inf(1)}},
		{[]float32{math.MaxFloat32}, 1, 1, []float64{math.MaxFloat64}},
	} {
		if values, err := ResamplePoly(t.Context(), nil, test.input, test.up, test.down, test.taps); err == nil || values != nil {
			t.Fatalf("accepted invalid resampling: %v", test)
		}
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := ResamplePoly(ctx, nil, []float32{1}, 1, 1, []float64{1}); err == nil {
		t.Fatal("accepted cancelled resampling")
	}
}
