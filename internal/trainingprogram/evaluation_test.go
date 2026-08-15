package trainingprogram

import (
	"math"
	"testing"
)

func TestNativeEvaluationMetrics(t *testing.T) {
	t.Parallel()
	tests := []struct {
		metric EvaluationMetric
		got    []float32
		want   []float32
		value  float64
	}{
		{MetricTokenAccuracy, []float32{1, 2, 4}, []float32{1, 2, 3}, 2.0 / 3},
		{MetricTableAccuracy, []float32{1, 0}, []float32{1, 1}, 0.5},
		{MetricForecastMAE, []float32{2, 4}, []float32{1, 2}, 1.5},
		{MetricAudioSNR, []float32{0.5, -0.5}, []float32{1, -1}, 6.020599913279624},
		{MetricImagePSNRSigned, []float32{0, 0}, []float32{1, -1}, 6.020599913279624},
		{MetricVideoPSNRSigned, []float32{0, 0}, []float32{1, -1}, 6.020599913279624},
		{MetricImagePSNRUnit, []float32{0, 0}, []float32{1, -1}, 0},
		{MetricVideoPSNRUnit, []float32{0, 0}, []float32{1, -1}, 0},
	}
	for _, test := range tests {
		got, err := EvaluateNative(test.metric, test.got, test.want)
		if err != nil || math.Abs(got-test.value) > 1e-12 {
			t.Fatalf("metric %s=%g, want %g: %v", test.metric, got, test.value, err)
		}
	}
}
