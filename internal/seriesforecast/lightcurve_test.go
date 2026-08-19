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

func TestLightCurveTrainingProcessorReservesTwoPatches(t *testing.T) {
	// Six observations of band 500 at patch 2, horizon 3: the evaluation
	// pairing would keep a single-patch context of 3; the training pairing
	// reserves two patches and shortens the target instead.
	record := trainingdata.RawRecord{ID: "curve", Group: "train", Data: []byte(`{
		"objid": 7,
		"x": [1, 500, 11, 1, 2, 500, 12, 1, 3, 500, 13, 1, 4, 500, 14, 1, 5, 500, 15, 1, 6, 500, 16, 1]
	}`)}
	processor, err := LightCurveTrainingProcessor(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	example, err := processor(context.Background(), record)
	if err != nil {
		t.Fatal(err)
	}
	input, target, err := TrainingPair(example)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := input, []float32{11, 12, 13, 14}; !slices.Equal(got, want) {
		t.Fatalf("context=%v, want %v", got, want)
	}
	if got, want := target, []float32{15, 16}; !slices.Equal(got, want) {
		t.Fatalf("target=%v, want %v", got, want)
	}

	// A curve that cannot cover two patches plus one target point refuses.
	short := trainingdata.RawRecord{ID: "short", Group: "train", Data: []byte(`{
		"objid": 8,
		"x": [1, 500, 11, 1, 2, 500, 12, 1, 3, 500, 13, 1, 4, 500, 14, 1]
	}`)}
	if _, err := processor(context.Background(), short); err == nil {
		t.Fatal("four-sample curve was not refused at the two-patch floor")
	}
}
