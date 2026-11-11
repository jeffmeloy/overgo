//go:build windows && modeltest

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/vitencoder"
)

// TestARenderedClipIsAnMP4TheJudgeReads holds a render to the format a person's request is given: the Wan four-step
// student draws two VidGen captions at its suite settings, 81 frames at 832 x 480, and each clip is an MP4 named by the
// digest of its bytes, the render's manifest naming "mp4" and, for each draw, the FFmpeg that wrote it. The judge's
// runtime binds each as it binds any rendered clip (verifyRendered) and reads it through FFmpeg, named by its bytes
// (videoDecode), its 81 frames at the clip's size; the frame-to-frame measure reads both through the suite's names. A
// pair screen of two formats is refused before it reads anything (TestARenderKeepsItsClipsInOneFormat, on the host).
// The judged run itself is left to a real comparison, so no reading of a two-clip suite joins the store's quality
// cells.
func TestARenderedClipIsAnMP4TheJudgeReads(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	tool := testcheck.Must(artifact.IdentifyFile(artifact.KindFile, ffmpeg))(t)
	const clips, seed, width, height = 2, 17, 832, 480
	items := testcheck.Must(evaluation.PickVideoCaptions(filepath.Join(roots.Datasets, "VidGen", "VidGen_1M_video_caption.json"), clips, seed))(t)
	prompts := make([]renderPrompt, len(items))
	for index, item := range items {
		prompts[index] = renderPrompt{ID: item.ID, Prompt: item.Caption}
	}
	drawn := t.TempDir()
	settings := map[string]float64{"width": width, "height": height, "frames": motionFrames}
	keys := testcheck.Must(renderMedia(ctx, roots.Store, roots.ResolveModelPath("Self-Forcing"), recipe.TaskVideoGen, drawn, prompts, settings, "", "render-mp4-test"))(t)
	var manifest renderManifest
	testcheck.NoError(t, jsonfile.Decode(filepath.Join(drawn, renderManifestFile), &manifest))
	if manifest.clipFormat() != "mp4" {
		t.Fatalf("the render's manifest names clips written as %q", manifest.clipFormat())
	}
	store := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	for _, item := range items {
		held := testcheck.Must(verifyRendered(ctx, store, drawn, manifest.Model, manifest.Recipe, keys[item.ID]))(t)
		path := filepath.Join(drawn, held.File)
		data := testcheck.Must(os.ReadFile(path))(t)
		named := testcheck.Must(artifact.IdentifyBytes(artifact.KindOutput, data))(t).DigestHex() + drawnExtensions[media.MP4MediaType]
		if held.File != named || !media.OpensAsMP4(data) || held.Encoding.Tool != tool || !strings.Contains(strings.Join(held.Encoding.Args, " "), "libx264") {
			t.Fatalf("%s drew %s (%d bytes, written by %+v), want an MP4 named %s written by FFmpeg %s", item.ID, held.File, len(data), held.Encoding, named, tool)
		}
		decode := testcheck.Must(videoDecode(path, "", motionFrameRate, motionFrames))(t)
		frames := testcheck.Must(media.DecodeVideoFile(ctx, path, ffmpeg, decode.FPS, decode.Frames))(t)
		if decode.Format != "ffmpeg" || decode.Tool != tool || len(frames) != motionFrames || frames[0].Bounds().Dx() != width || frames[0].Bounds().Dy() != height {
			t.Fatalf("the judge reads %s as %+v: %d frames of %v, want FFmpeg %s and %d frames of %dx%d", held.File, decode, len(frames), frames[0].Bounds(), tool, motionFrames, width, height)
		}
	}

	suite := testcheck.Must(evaluation.BuildGeneratedVideoMatching(items, nil, keys, "judge"))(t)
	encoder := testcheck.Must(vitencoder.Load(roots.ResolveModelPath("webssl-dino300m-full2b-224")))(t)
	card := testcheck.Must(encoder.OpenCard(ctx, 0))(t)
	t.Cleanup(func() { testcheck.NoError(t, card.Close()) })
	testcheck.NoError(t, measureMotion(ctx, card, "webssl-dino300m-full2b-224", &suite, drawn, ""))
	if suite.Motion == nil || len(suite.Motion.Clips) != clips {
		t.Fatalf("the frame-to-frame measure read %+v, want both clips", suite.Motion)
	}
	for key, measured := range suite.Motion.Clips {
		t.Logf("%s: mean frame-to-frame similarity %.4f, lowest %.4f", key, measured.Mean, measured.Lowest)
	}
}
