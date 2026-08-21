package trainingprogram

import (
	"errors"
	"math"
)

// EvaluateNative applies the objective-declared output metric.
func EvaluateNative(metric EvaluationMetric, prediction, target []float32) (float64, error) {
	if len(prediction) == 0 || len(prediction) != len(target) {
		return 0, errors.New("training evaluation: value geometry differs")
	}
	var squared, signal, absolute float64
	var correct int
	for index, predicted := range prediction {
		want := target[index]
		if !finite(float64(predicted)) || !finite(float64(want)) {
			return 0, errors.New("training evaluation: non-finite value")
		}
		delta := float64(predicted - want)
		squared += delta * delta
		signal += float64(want) * float64(want)
		absolute += math.Abs(delta)
		if predicted == want {
			correct++
		}
	}
	count := float64(len(prediction))
	switch metric {
	case MetricTokenAccuracy, MetricTableAccuracy:
		return float64(correct) / count, nil
	case MetricImagePSNRSigned, MetricVideoPSNRSigned:
		mse := squared / count
		if mse == 0 {
			return math.Inf(1), nil
		}
		return 10 * math.Log10(4/mse), nil
	case MetricImagePSNRUnit, MetricVideoPSNRUnit:
		mse := squared / count
		if mse == 0 {
			return math.Inf(1), nil
		}
		return 10 * math.Log10(1/mse), nil
	case MetricAudioSNR:
		if squared == 0 {
			return math.Inf(1), nil
		}
		if signal == 0 {
			return math.Inf(-1), nil
		}
		return 10 * math.Log10(signal/squared), nil
	case MetricForecastMAE:
		return absolute / count, nil
	default:
		return 0, errors.New("training evaluation: metric unsupported")
	}
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
