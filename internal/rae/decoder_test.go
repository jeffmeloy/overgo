package rae

import (
	"math"
	"testing"

	"overgo/internal/fixtureasset"
)

// referenceTensor is one golden tensor the fixture manifest pins.
type referenceTensor struct {
	File   string `json:"file"`
	Shape  []int  `json:"shape"`
	SHA256 string `json:"sha256"`
}

// decoderReference is fixtures/rae/decoder.json: a seeded normalized latent
// and the image the reference decoder made of it.
type decoderReference struct {
	Source     string          `json:"source"`
	Commit     string          `json:"reference_commit"`
	Generator  string          `json:"generator_sha256"`
	Encoder    string          `json:"encoder"`
	Decoder    string          `json:"decoder"`
	Statistics string          `json:"statistics"`
	Seed       int             `json:"seed"`
	Latent     referenceTensor `json:"latent"`
	Image      referenceTensor `json:"image"`
}

func loadReference(t *testing.T, directory string, tensor referenceTensor) []float32 {
	t.Helper()
	elements := 1
	for _, extent := range tensor.Shape {
		elements *= extent
	}
	values, err := fixtureasset.LoadF32(directory, tensor.File, tensor.SHA256, elements)
	if err != nil {
		t.Fatal(err)
	}
	return values
}

// compareReference is the largest difference of got from the reference's
// values and the largest magnitude the reference holds.
func compareReference(t *testing.T, got, want []float32) (worst, scale float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("produced %d values, the reference has %d", len(got), len(want))
	}
	for index := range got {
		worst = max(worst, math.Abs(float64(got[index]-want[index])))
		scale = max(scale, math.Abs(float64(want[index])))
	}
	return worst, scale
}
