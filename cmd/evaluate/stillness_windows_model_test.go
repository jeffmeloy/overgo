//go:build windows && modeltest

package main

import (
	"image"
	"os"
	"path/filepath"
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

// stillnessClips is how many VidGen clips the stillness fixture draws: the last of them is frozen.
const stillnessClips = 4

// TestAFrozenClipReadsAsTooStill holds the frame-to-frame measure to a clip that does not move: judge-calibration/
// sensitivity measured the judge's caption questions blind to a clip frozen on its first frame (1 lower of 10), and the
// measure read such a clip's similarity, 1 between every frame, as the best a clip has. Of four VidGen clips, three are
// asked about as they are and the fourth frozen on its first frame, each measured beside its source: the frozen clip
// is more alike frame to frame than the stillest of the four sources, and it alone, so a quarter of the suite's clips
// read as stiller than its sources.
func TestAFrozenClipReadsAsTooStill(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	sources := filepath.Join(roots.Datasets, "VidGen")
	listed := testcheck.Must(os.ReadDir(filepath.Join(roots.Datasets, "VidGen_frames")))(t)
	drawn := t.TempDir()
	var items []evaluation.VideoCaptionItem
	names := map[string]string{}
	for _, entry := range listed {
		clip := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
		frames, err := media.DecodeVideoFile(ctx, filepath.Join(sources, clip+clipFileExtension), ffmpeg, motionFrameRate, motionFrames)
		if err != nil || len(frames) < frameHalf {
			continue
		}
		name := clip + clipFileExtension
		if len(items) == stillnessClips-1 {
			name = clip + ".gif"
			frozen := make([]image.Image, len(frames))
			for index := range frozen {
				frozen[index] = frames[0]
			}
			writeCalibrationClip(t, ctx, filepath.Join(drawn, name), frozen)
		} else {
			testcheck.NoError(t, os.WriteFile(filepath.Join(drawn, name), testcheck.Must(os.ReadFile(filepath.Join(sources, name)))(t), frameFileMode))
		}
		items = append(items, evaluation.VideoCaptionItem{ID: clip, Caption: "a clip", Questions: []evaluation.VideoCaptionQuestion{{Options: []string{"a clip", "another"}, Answer: 0}}})
		names[clip] = name
		if len(items) == stillnessClips {
			break
		}
	}
	if len(items) < stillnessClips {
		t.Fatalf("VidGen holds %d clips of two sampled frames or more among VidGen_frames' names, not %d", len(items), stillnessClips)
	}
	suite := testcheck.Must(evaluation.BuildGeneratedVideoMatching(items, nil, names, "judge"))(t)
	encoder := testcheck.Must(vitencoder.Load(roots.ResolveModelPath("webssl-dino300m-full2b-224")))(t)
	card := testcheck.Must(encoder.OpenCard(ctx, 0))(t)
	t.Cleanup(func() { testcheck.NoError(t, card.Close()) })
	testcheck.NoError(t, measureMotion(ctx, card, "webssl-dino300m-full2b-224", &suite, drawn, sources))
	var ceiling float64
	for _, source := range suite.Motion.Sources {
		ceiling = max(ceiling, source.Mean)
	}
	frozen := items[stillnessClips-1].ID
	for _, item := range items {
		t.Logf("%s: mean %.4f beside its source's %.4f, the stillest source %.4f", item.ID, suite.Motion.Clips[item.ID].Mean, suite.Motion.Sources[item.ID].Mean, ceiling)
		if stiller := suite.Motion.Clips[item.ID].Mean > ceiling; stiller != (item.ID == frozen) {
			t.Errorf("%s reads as stiller than the stillest source: %t, frozen: %t", item.ID, stiller, item.ID == frozen)
		}
	}
	metrics := map[string]float64{}
	for _, metric := range suite.Motion.Metrics() {
		metrics[metric.Name] = metric.Value
	}
	// The metric is read by the name a judged reading records it under, which stored readings keep.
	if still, read := metrics["stiller-than-sources"]; !read || still != 1.0/stillnessClips {
		t.Fatalf("the suite records %v, want a quarter of its clips stiller than its sources", metrics)
	}
}
