package hostoptimizer

import (
	"math"
	"testing"
)

func TestMuonFlatOptimizerFixture(t *testing.T) {
	const (
		MatrixElements  = 4
		parameterCount  = 7
		paritySteps     = 4
		gradientDivisor = 64
		weightTolerance = 2e-7
	)
	weights := []float32{0.5, -0.25, 0.125, -0.75, 0.25, -0.5, 0.75}
	gradients := make([]float32, parameterCount)
	plan := mustPlan(t, parameterCount, []GroupSpec{
		{Name: "matrix", Start: 0, End: MatrixElements, Rows: 2, Cols: 2},
		{Name: "vector", Start: MatrixElements, End: parameterCount, Rows: 1, Cols: 3},
	})
	instance := mustOptimizer(t, weights, gradients, plan, fixtureConfig)
	for step := range paritySteps {
		for index := range parameterCount {
			gradients[index] = float32((step+1)*(index+1)) / gradientDivisor
		}
		instance.Step()
	}
	wantWeights := []float32{0.506821036, -0.261363506, 0.113637798, -0.756814361, 0.242264956, -0.509281993, 0.739170909}
	wantMomentum := []float64{0.14863085937499998, 0.29726171874999996, 0.44589257812499999, 0.59452343749999992, 0.74315429687500001, 0.89178515624999999, 1.040416015625}
	for index := range parameterCount {
		if difference := math.Abs(float64(weights[index] - wantWeights[index])); difference > weightTolerance {
			t.Fatalf("weight[%d] difference = %g: got %.9g want %.9g", index, difference, weights[index], wantWeights[index])
		}
	}
	state := instance.Snapshot()
	for index := range parameterCount {
		if state.Momentum[index] != wantMomentum[index] {
			t.Fatalf("momentum[%d] = %.17g, want %.17g", index, state.Momentum[index], wantMomentum[index])
		}
	}
}
