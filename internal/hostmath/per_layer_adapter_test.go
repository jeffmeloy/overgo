package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

func TestPerLayerAdapterBackwardFiniteDifference(t *testing.T) {
	const rows, hidden, width = 2, 3, 2
	rng := rand.New(rand.NewSource(17))
	values := func(n int) []float32 {
		result := make([]float32, n)
		for index := range result {
			result[index] = float32(rng.NormFloat64() * 0.2)
		}
		return result
	}
	input, side := values(rows*hidden), values(rows*width)
	gate, projection, norm := values(width*hidden), values(hidden*width), values(hidden)
	for index := range norm {
		norm[index] += 1
	}
	seed := values(rows * hidden)
	output, trace, err := PerLayerAdapterForward(input, side, gate, projection, norm, rows, hidden, width, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	_ = output
	dInput, dSide, dGate, dProjection, dNorm, err := PerLayerAdapterBackward(
		input, side, gate, projection, norm, seed, rows, hidden, width, 1e-6, trace,
	)
	if err != nil {
		t.Fatal(err)
	}
	loss := func() float64 {
		got, _, err := PerLayerAdapterForward(input, side, gate, projection, norm, rows, hidden, width, 1e-6)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for index := range got {
			sum += float64(got[index]) * float64(seed[index])
		}
		return sum
	}
	for _, item := range []struct {
		name        string
		value, grad []float32
	}{
		{"input", input, dInput}, {"side", side, dSide}, {"gate", gate, dGate},
		{"projection", projection, dProjection}, {"norm", norm, dNorm},
	} {
		for index := range item.value {
			const step = 1e-3
			original := item.value[index]
			item.value[index] = original + step
			positive := loss()
			item.value[index] = original - step
			negative := loss()
			item.value[index] = original
			want := (positive - negative) / (2 * step)
			if delta := math.Abs(float64(item.grad[index]) - want); delta > 3e-3 {
				t.Fatalf("%s[%d] grad=%g want=%g delta=%g", item.name, index, item.grad[index], want, delta)
			}
		}
	}
}
