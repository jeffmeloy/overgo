package trainingprogram

import (
	"math"
	"strings"
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
		{MetricTableMSE, []float32{2, 4}, []float32{1, 2}, 2.5},
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

// TestNativeEvaluationRefusesARatioThatHasNoValue holds that a ratio in
// decibels is refused, with the reason, where it has no finite value: each
// ratio metric refuses an answer equal to its target, saying the two match
// exactly, and the audio ratio refuses a silent target answered with sound,
// saying the target is silent. Neither comes back as an infinity with no
// error, which a caller would have to recognise for itself.
func TestNativeEvaluationRefusesARatioThatHasNoValue(t *testing.T) {
	t.Parallel()
	answer := []float32{0.5, -0.25}
	for _, metric := range []EvaluationMetric{MetricImagePSNRSigned, MetricVideoPSNRSigned, MetricImagePSNRUnit, MetricVideoPSNRUnit, MetricAudioSNR} {
		if value, err := EvaluateNative(metric, answer, answer); err == nil || !strings.Contains(err.Error(), "exactly") {
			t.Errorf("metric %s on an answer equal to its target = (%g, %v), want a refusal saying they match exactly", metric, value, err)
		}
	}
	if value, err := EvaluateNative(MetricAudioSNR, []float32{1, 0}, []float32{0, 0}); err == nil || !strings.Contains(err.Error(), "silent") {
		t.Errorf("metric %s on a silent target answered with sound = (%g, %v), want a refusal saying the target is silent", MetricAudioSNR, value, err)
	}
}
