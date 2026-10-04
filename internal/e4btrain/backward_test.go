package e4btrain

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// Finite-difference bounds: central differences along one random direction
// per tensor, at a step the f32 forward resolves.
const (
	finiteStep     = 1e-2
	finiteRelative = 5e-3
	finiteFloor    = 2e-4
)

// TestE4BHostGradientsMatchFiniteDifference holds the host kit's composed
// full-parameter gradient to the loss it differentiates: for every trained
// tensor of the tiny E4B-shaped decoder, the directional derivative along a
// random direction matches central differences of the loss.
func TestE4BHostGradientsMatchFiniteDifference(t *testing.T) {
	t.Parallel()
	m := tinyModel(t, 1)
	loss, err := m.differentiate(hostAttention)
	if err != nil {
		t.Fatal(err)
	}
	evaluated, err := m.Loss()
	if err != nil {
		t.Fatal(err)
	}
	if loss != evaluated || !(loss > 0) {
		t.Fatalf("loss through the gradient pass %v, evaluated %v", loss, evaluated)
	}
	gradients := slices.Clone(m.gradients)
	rng := rand.New(rand.NewSource(2))
	for _, group := range m.plan.Groups() {
		original := slices.Clone(m.weights[group.Start:group.End])
		direction := make([]float64, len(original))
		var norm, analytic float64
		for index := range direction {
			direction[index] = rng.NormFloat64()
			norm += direction[index] * direction[index]
		}
		for index := range direction {
			direction[index] /= math.Sqrt(norm)
			analytic += direction[index] * float64(gradients[group.Start+index])
		}
		lossAt := func(step float64) float64 {
			values := m.weights[group.Start:group.End]
			for index := range values {
				values[index] = float32(float64(original[index]) + step*direction[index])
			}
			value, err := m.Loss()
			copy(values, original)
			if err != nil {
				t.Fatal(err)
			}
			return value
		}
		central := func(step float64) float64 { return (lossAt(step) - lossAt(-step)) / (2 * step) }
		// Richardson's extrapolation cancels the central difference's
		// second-order error, which the soft cap's curvature makes large.
		numeric := (4*central(finiteStep/2) - central(finiteStep)) / 3
		bound := finiteRelative*max(math.Abs(numeric), math.Abs(analytic)) + finiteFloor
		if difference := math.Abs(numeric - analytic); difference > bound || analytic == 0 {
			t.Errorf("%s: analytic %.6g, finite difference %.6g (|difference| %.3g > %.3g)", group.Name, analytic, numeric, difference, bound)
		}
	}
}
