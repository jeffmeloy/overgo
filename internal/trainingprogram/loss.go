package trainingprogram

import "errors"

// MeanSquaredErrorF32 computes mean squared error and optionally its prediction
// gradient using float64 accumulation.
func MeanSquaredErrorF32(prediction, target []float32, withGradient bool) (float64, []float32, error) {
	if len(prediction) == 0 || len(prediction) != len(target) {
		return 0, nil, errors.New("training program: MSE storage differs")
	}
	inverse := 1 / float64(len(target))
	var loss float64
	var gradient []float32
	if withGradient {
		gradient = make([]float32, len(target))
	}
	for index := range target {
		difference := float64(prediction[index]) - float64(target[index])
		loss += difference * difference * inverse
		if withGradient {
			gradient[index] = float32(2 * difference * inverse)
		}
	}
	return loss, gradient, nil
}
