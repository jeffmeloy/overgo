//go:build windows && modeltest

package main

import (
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/vitencoder"
)

// TestAClipRecordsItsFrameToFrameSimilarity holds the clip measure to what it is for. A VidGen source clip read at 16
// frames a second, as a Wan clip is, is measured three ways: as it is; with one frame replaced by another clip's, a
// cut no caption describes; and with one frame repeating the clip's first, a chunk that started over. Each fault lowers
// the similarity beside the frame it struck below the clean clip's there, and the cut, larger than any step real motion
// takes, is the clip's lowest. (On 2026-11-08 the first clip's own lowest, 0.514, was a sharp change near its end, so
// a repeat of its first frame, which differs from frame 39 less than that, shows beside frame 40 and not as the
// clip's lowest: a real clip's lowest is the floor its generated clips are read against.) Then a suite asking about the
// clean clip, its source the same clip, records its motion: no clip falls below its source.
func TestAClipRecordsItsFrameToFrameSimilarity(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	sources := filepath.Join(roots.Datasets, "VidGen")
	listed := testcheck.Must(os.ReadDir(filepath.Join(roots.Datasets, "VidGen_frames")))(t)
	var clips []string
	for _, entry := range listed {
		clips = append(clips, strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name())))
	}
	if len(clips) < 2 {
		t.Fatalf("VidGen_frames names %d clips, want two", len(clips))
	}
	encoder := testcheck.Must(vitencoder.Load(roots.ResolveModelPath("webssl-dino300m-full2b-224")))(t)
	card := testcheck.Must(encoder.OpenCard(ctx, 0))(t)
	t.Cleanup(func() { testcheck.NoError(t, card.Close()) })
	frames := testcheck.Must(media.DecodeVideoFile(ctx, filepath.Join(sources, clips[0]+clipFileExtension), ffmpeg, motionFrameRate, motionFrames))(t)
	other := testcheck.Must(media.DecodeVideoFile(ctx, filepath.Join(sources, clips[1]+clipFileExtension), ffmpeg, motionFrameRate, motionFrames))(t)
	struck := len(frames) / 2
	replaced, repeated := slices.Clone(frames), slices.Clone(frames)
	replaced[struck], repeated[struck] = other[0], frames[0]
	clean := testcheck.Must(evaluation.NewFrameSimilarity(testcheck.Must(embedFrames(ctx, card, frames))(t)))(t)
	t.Logf("%s: %d frames, mean similarity %.4f, lowest %.4f", clips[0], len(frames), clean.Mean, clean.Lowest)
	// beside is the lower similarity of the struck frame to its two neighbours.
	beside := func(measured evaluation.FrameSimilarity) float64 {
		return min(measured.Neighbours[struck-1], measured.Neighbours[struck])
	}
	for _, fault := range []struct {
		name   string
		frames []image.Image
		lowest bool
	}{{"one frame replaced by another clip's", replaced, true}, {"one frame repeating the first", repeated, false}} {
		measured := testcheck.Must(evaluation.NewFrameSimilarity(testcheck.Must(embedFrames(ctx, card, fault.frames))(t)))(t)
		lowestAt := slices.Index(measured.Neighbours, measured.Lowest)
		t.Logf("%s at frame %d: %.4f beside it, %.4f in the clean clip; mean %.4f, lowest %.4f between frames %d and %d",
			fault.name, struck, beside(measured), beside(clean), measured.Mean, measured.Lowest, lowestAt, lowestAt+1)
		if beside(measured) >= beside(clean) {
			t.Errorf("a clip with %s at frame %d reads %.4f beside it, want below the clean clip's %.4f", fault.name, struck, beside(measured), beside(clean))
		}
		if fault.lowest && (measured.Lowest >= clean.Lowest || measured.Lowest != beside(measured)) {
			t.Errorf("a clip with %s at frame %d has its lowest similarity %.4f between frames %d and %d, want the one beside that frame, below the clean clip's %.4f",
				fault.name, struck, measured.Lowest, lowestAt, lowestAt+1, clean.Lowest)
		}
	}

	drawn := t.TempDir()
	name := clips[0] + clipFileExtension
	testcheck.NoError(t, os.WriteFile(filepath.Join(drawn, name), testcheck.Must(os.ReadFile(filepath.Join(sources, name)))(t), frameFileMode))
	suite := testcheck.Must(evaluation.BuildGeneratedVideoMatching([]evaluation.VideoCaptionItem{{
		ID: clips[0], Caption: "a clip", Questions: []evaluation.VideoCaptionQuestion{{Options: []string{"a clip", "another"}, Answer: 0}},
	}}, nil, map[string]string{clips[0]: name}, "judge"))(t)
	testcheck.NoError(t, measureMotion(ctx, card, "webssl-dino300m-full2b-224", &suite, drawn, sources))
	metrics := map[string]float64{}
	for _, metric := range suite.Motion.Metrics() {
		metrics[metric.Name] = metric.Value
	}
	t.Logf("the suite records %v", metrics)
	// The same frames embed again on the card, where a sum's order may differ in its last bits.
	const embeddedAgain = 1e-5
	if below, sourced := metrics[evaluation.BelowSourceMetric]; !sourced || below != 0 || math.Abs(metrics[evaluation.LowestFrameSimilarityMetric]-clean.Lowest) > embeddedAgain {
		t.Errorf("a suite whose clip is its own source records %v, want its clip's lowest %.4f and none below its source", metrics, clean.Lowest)
	}
}
