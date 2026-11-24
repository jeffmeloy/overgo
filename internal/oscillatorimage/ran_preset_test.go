package oscillatorimage

import (
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// TestUn0OutputsRecordWhatTheyRan holds image-presets for Un-0: a picture and a clip it draws record the preset they
// ran, their size, a clip's frames, the oscillators' integration steps and their sampler, so the scorecard shows their
// served time with what it is a time of; the full measurement of 2026-11-24 left Un-0's first outputs unshown, "its
// measured attempt recorded no preset". A clip's preset is no part of its published JSON, whose identity its frozen
// runs pin.
func TestUn0OutputsRecordWhatTheyRan(t *testing.T) {
	t.Parallel()
	model := loadArtifactModel(t)
	plan := testcheck.Must(model.prepare(t.Context(), Request{Class: 1, Seed: 11}))(t)
	features := testcheck.Must(model.integrate(t.Context(), plan))(t)
	image := testcheck.Must(model.decode(t.Context(), features))(t)
	want := media.Preset{Width: image.Width, Height: image.Height, Steps: model.Cfg.NumSteps, Sampler: "Kuramoto Euler"}
	if got := media.PresetOf(image); got != want || !got.Ran() {
		t.Fatalf("the picture records the preset %+v; want %+v, one that ran", got, want)
	}
	request := VideoRequest{Class: 1, Seed: 11, Frames: 6, Scale: 2}
	videoPlan := testcheck.Must(model.prepareVideo(t.Context(), request))(t)
	videoFeatures := testcheck.Must(model.integrateVideo(t.Context(), videoPlan))(t)
	clip := testcheck.Must(model.decodeVideo(t.Context(), videoFeatures))(t)
	want = media.Preset{Width: clip.Width, Height: clip.Height, Frames: request.Frames, Steps: model.Cfg.NumSteps, Sampler: "Kuramoto Euler"}
	if got := media.PresetOf(clip); got != want || !got.Ran() {
		t.Fatalf("the clip records the preset %+v; want %+v, one that ran", got, want)
	}
}
