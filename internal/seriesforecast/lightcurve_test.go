package seriesforecast

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/trainingdata"
)

func TestLightCurveProcessorDerivesBandAndBoundary(t *testing.T) {
	processor, err := LightCurveProcessor(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	record := trainingdata.RawRecord{ID: "curve", Group: "heldout", Data: []byte(`{
		"objid": 7,
		"x": [4, 500, 14, 1, 1, 600, 20, 1, 3, 500, 13, 1, 1, 500, 11, 1, 2, 500, 12, 1, 0, 0, 0, 0]
	}`)}
	example, err := processor(context.Background(), record)
	if err != nil {
		t.Fatal(err)
	}
	input, target, err := TrainingPair(example)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := input, []float32{11, 12}; !slices.Equal(got, want) {
		t.Fatalf("context=%v, want %v", got, want)
	}
	if got, want := target, []float32{13, 14}; !slices.Equal(got, want) {
		t.Fatalf("target=%v, want %v", got, want)
	}
}
