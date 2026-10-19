//go:build windows

package latentimage

import (
	"bytes"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/safetensors"
)

func TestMOSSImageGenerationContract(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	path := filepath.Join("..", "..", "fixtures", "moss", "generator-options.safetensors")
	id, err := artifact.IdentifyFile(artifact.KindFile, path)
	if err != nil || id.DigestHex() != "8489f99bbbca59394ebcaf90f2573fc50a7abaf8093cb6914fb398526d443608" {
		t.Fatal("native options identity", err)
	}
	source, err := safetensors.OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	var contract struct {
		Comparison struct {
			Tolerances map[string]mossStageTolerance `json:"dtype_tolerances"`
		}
	}
	if err = readJSON(filepath.Join("..", "..", "fixtures", "moss", "oracle-contract.json"), &contract); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name                       string
		guidance                   float32
		height, width, steps, rows int
	}{
		{"guidance-zero", 0, 32, 48, 3, 42},
		{"guidance-one", 1, 32, 48, 3, 32},
		{"guidance-fractional", 2.5, 32, 48, 3, 42},
		{"minimum", 3, 16, 16, 1, 26},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := Request{Prompt: "blue cube", NegativePrompt: "red", GuidanceScale: &tc.guidance, Height: tc.height, Width: tc.width, Steps: tc.steps, Seed: 42}
			var report bytes.Buffer
			if err := compareGeneration(t.Context(), root, tc.name, tc.name+".", request, id, source, contract.Comparison.Tolerances, &report); err != nil {
				t.Log(report.String())
				t.Fatal(err)
			}
			t.Log(report.String())
			if bytes.Count(report.Bytes(), []byte("\n")) != tc.rows {
				t.Fatal("native option boundary coverage")
			}
		})
	}
	for _, guidance := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		if err := ValidateRequest(Request{Prompt: "cube", Width: 16, Height: 16, GuidanceScale: &guidance}); err == nil {
			t.Fatal("invalid guidance accepted", guidance)
		}
	}
}
