package main

import (
	"encoding/json"
	"math"
	"testing"
)

func TestSummarizeProducesFiniteJSONStatistics(t *testing.T) {
	tests := []struct {
		name   string
		data   []float32
		finite bool
	}{
		{name: "empty", finite: true},
		{name: "constant", data: []float32{3, 3, 3}, finite: true},
		{name: "non-finite", data: []float32{1, float32(math.NaN())}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			summary := summarize(test.data)
			if summary.Finite != test.finite || summary.Std != 0 ||
				math.IsInf(summary.Min, 0) || math.IsInf(summary.Max, 0) {
				t.Fatalf("summary = %+v", summary)
			}
			if _, err := json.Marshal(summary); err != nil {
				t.Fatal(err)
			}
		})
	}
}
