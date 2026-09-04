package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestAudioFrontendCommandPublishesBoundaryInvariantEvidence(t *testing.T) {
	root := t.TempDir()
	samples := []int16{1000, -2000, 3000, -4000, 5000, -6000, 7000, -8000}
	input := filepath.Join(root, "input.wav")
	if err := os.WriteFile(input, testutil.MonoPCM16WAV(8, samples), 0o600); err != nil {
		t.Fatal(err)
	}
	policy := dataset.AudioInspectionPolicy{MaximumEncodedBytes: 1024, MaximumSamples: 8, ClipThreshold: 1, Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1, MaximumAbsoluteDCOffset: 1}}
	config := audiodsp.FrontendConfig{SampleRate: 8, Geometry: recipecontract.AudioFrameGeometry{WindowSamples: 4, HopSamples: 2, FeatureBins: 2}, FFTLength: 4, FrameSpan: 4, PadLeft: 2, PadRight: 2, Padding: "reflect", Window: "rectangular", Mel: audiodsp.MelConfig{Scale: "htk", MaxFrequency: 4}, Log: audiodsp.LogConfig{Base: "natural", GuardMode: "add", Guard: 1e-6, Scale: 1}}
	paths := []string{filepath.Join(root, "policy.json"), filepath.Join(root, "frontend.json")}
	for i, value := range []any{policy, config} {
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(paths[i], data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var first frontendObservation
	for chunk := 1; chunk <= len(samples); chunk++ {
		args := []string{"-repo", filepath.Join(root, "store"), "-input", input, "-policy", paths[0], "-frontend", paths[1], "-frontend-memory", "1048576", "-chunk-samples", strconv.Itoa(chunk), "-reconstruct"}
		var output bytes.Buffer
		if err := run(t.Context(), args, &output); err != nil {
			t.Fatalf("chunk %d: %v", chunk, err)
		}
		var observation frontendObservation
		if err := json.Unmarshal(bytes.Split(output.Bytes(), []byte{'\n'})[0], &observation); err != nil {
			t.Fatal(err)
		}
		if observation.Frames != 5 || observation.Values != 10 || observation.SourceMaximumError == nil || *observation.SourceMaximumError != 0 || observation.ReconstructionSamples != 8 {
			t.Fatalf("observation=%+v", observation)
		}
		if chunk == 1 {
			first = observation
		} else if first.Config != observation.Config || first.FeaturesSHA256 != observation.FeaturesSHA256 || first.ReconstructionSHA256 != observation.ReconstructionSHA256 {
			t.Fatalf("chunk %d changed evidence", chunk)
		}
	}
}
