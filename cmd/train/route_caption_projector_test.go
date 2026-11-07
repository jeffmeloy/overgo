package main

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestHeldSourcesStayOutOfTraining holds the caption projector's held-back share to whole source videos: every sample
// of the first held sources in the capture's order is scored and none is trained on, a later sample of a held source
// with them, and every other sample is trained on in its order; holding none trains on every sample.
func TestHeldSourcesStayOutOfTraining(t *testing.T) {
	t.Parallel()
	sources := []string{"a", "b", "a", "c", "b", "d"}
	samples, sequences := make([]evaluation.CapturedCaption, len(sources)), make([]captionSequence, len(sources))
	for index, source := range sources {
		samples[index].Source, sequences[index].vid = source, strconv.Itoa(index)
	}
	vids := func(chosen []captionSequence) []string {
		names := make([]string, len(chosen))
		for index, sequence := range chosen {
			names[index] = sequence.vid
		}
		return names
	}
	trained, scored := heldBack(samples, sequences, 2)
	if !slices.Equal(vids(scored), []string{"0", "1", "2", "4"}) || !slices.Equal(vids(trained), []string{"3", "5"}) {
		t.Fatalf("holding two sources scored %v and trained %v; want every sample of a and b scored, the rest trained", vids(scored), vids(trained))
	}
	if trained, scored := heldBack(samples, sequences, 0); len(scored) != 0 || len(trained) != len(sources) {
		t.Fatalf("holding no source scored %d samples and trained %d; want every sample trained", len(scored), len(trained))
	}
}

// TestAMeanStartPlacesTheTrainedCaptionsMean writes a capture of three samples from two sources, the first source held
// back: the mean start's offsets are the mean of the trained source's captions alone and its weights zero, the zero
// start is all zero, and any other start is refused.
func TestAMeanStartPlacesTheTrainedCaptionsMean(t *testing.T) {
	t.Parallel()
	const encoderWidth, textWidth = 2, 3
	capture := evaluation.CaptionTrainingCapture{EncoderWidth: encoderWidth, TextWidth: textWidth}
	for index, source := range []string{"held", "trained", "trained"} {
		capture.Samples = append(capture.Samples, evaluation.CapturedCaption{
			CaptionTrainingClip: evaluation.CaptionTrainingClip{Vid: strconv.Itoa(index), Source: source}, Frame: testutil.ArtifactID(t, artifact.KindFile, strconv.Itoa(index)),
		})
	}
	representations := filepath.Join(t.TempDir(), "capture.safetensors")
	targets := []float32{100, 100, 100, 1, 2, 3, 3, 4, 5}
	testcheck.NoError(t, safetensors.Save(representations, map[string][]float32{evaluation.CaptureTarget: targets},
		map[string][]int{evaluation.CaptureTarget: {len(capture.Samples), textWidth}}, nil))
	samples := representations + evaluation.CaptureSamplesSuffix
	mean := testcheck.Must(captionStart(captionMeanStart, samples, capture, 1))(t)
	if !slices.Equal(mean.Bias, []float32{2, 3, 4}) || len(mean.Weight) != textWidth*encoderWidth || slices.ContainsFunc(mean.Weight, func(value float32) bool { return value != 0 }) {
		t.Fatalf("the mean start is %v and %v; want offsets the trained captions' mean [2 3 4] and zero weights", mean.Bias, mean.Weight)
	}
	zero := testcheck.Must(captionStart(captionZeroStart, samples, capture, 1))(t)
	if slices.ContainsFunc(append(zero.Weight, zero.Bias...), func(value float32) bool { return value != 0 }) {
		t.Fatalf("the zero start is %v and %v; want every value zero", zero.Weight, zero.Bias)
	}
	if _, err := captionStart("fit", samples, capture, 1); err == nil {
		t.Fatal("a start the route does not name was not refused")
	}
	// A run that holds back every source is refused before it writes a start or opens anything.
	testcheck.NoError(t, jsonfile.Write(samples, capture, 0o600))
	from := filepath.Join(t.TempDir(), "start.safetensors")
	err := runRoute(t.Context(), []string{"caption-projector", "-capture", samples, "-placeholder", "<|image_pad|>", "-from", from, "-start", captionMeanStart,
		"-out", filepath.Join(t.TempDir(), "out"), "-steps", "1", "-batch", "1", "-held", "2", "-store", filepath.Join(t.TempDir(), "store")})
	if _, statErr := os.Stat(from); err == nil || !strings.Contains(err.Error(), "fewer than a batch") || !errors.Is(statErr, fs.ErrNotExist) {
		t.Fatalf("a run holding back every source was refused for %v and left its start written (%v); want the refusal and no start", err, statErr)
	}
}
