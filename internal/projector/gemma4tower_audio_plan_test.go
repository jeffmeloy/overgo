package projector

import "testing"

func TestGemma4TowerAudioPlanReused(t *testing.T) {
	spec := Gemma4AudioTowerSpec{
		MelBins: 4, FFTLength: 8, FrameLength: 4, HopLength: 2, SampleRate: 8,
		MinFrequency: 0, MaxFrequency: 4, MelFloor: 1e-6,
	}
	plan := newGemma4AudioFrontendPlan(spec)
	window := &plan.window[0]
	basis := &plan.cosine[0]
	filterbank := &plan.filterbank[0]
	samples := []float32{0, 0.25, -0.25, 0.5, -0.5, 0.25}
	first, firstFrames, err := plan.preprocess(samples, spec.SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	second, secondFrames, err := plan.preprocess(samples, spec.SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	if window != &plan.window[0] || basis != &plan.cosine[0] || filterbank != &plan.filterbank[0] {
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
}
