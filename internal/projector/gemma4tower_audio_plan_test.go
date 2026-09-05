package projector

import (
	"testing"
)

func TestGemma4TowerAudioPlanReused(t *testing.T) {
	spec := Gemma4AudioTowerSpec{
		MelBins: 4, FFTLength: 8, FrameLength: 4, HopLength: 2, SampleRate: 8,
		MinFrequency: 0, MaxFrequency: 4, MelFloor: 1e-6,
	}
	plan := newAudioFrontendPlan(spec)
	frontend := plan.frontend
	samples := []float32{0, 0.25, -0.25, 0.5, -0.5, 0.25}
	first, firstFrames, err := plan.preprocess(t.Context(), samples, spec.SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	second, secondFrames, err := plan.preprocess(t.Context(), samples, spec.SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	if frontend != plan.frontend {
		t.Fatal("frontend tables were replaced")
	}
	if firstFrames != secondFrames || len(first) != len(second) {
		t.Fatalf("results=%d/%d frames=%d/%d", len(first), len(second), firstFrames, secondFrames)
	}
	for index := range first {
		if first[index] != second[index] {
			t.Fatalf("feature[%d]=%g/%g", index, first[index], second[index])
		}
	}
	// Captured from retained baseline 19ceb5d9 before replacing its DFT frontend.
	want := []float32{-2.37054, -2.3682508, -2.3659716, -2.3637023, -2.702603, -1.8235614, -1.3091679, -0.9330307}
	if firstFrames != 2 || len(first) != len(want) {
		t.Fatalf("geometry=%d/%d", firstFrames, len(first))
	}
	for index, value := range want {
		if first[index] != value {
			t.Fatalf("retained feature[%d]=%g want %g", index, first[index], value)
		}
	}
}
