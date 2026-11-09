//go:build windows && modeltest

package latentvideo

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The Wan student clip a finish is measured on: one the Self-Forcing student drew for the VidGen suite, 81 frames at
// 832 x 480 and 16 frames a second, finished at four times to 3328 x 1920.
const (
	studentClip       = "03fe55e239eaa030d1d347ad1bec26f3a0bcffeba17ad2551eaaf670bc76d163.gif"
	studentClipFrames = 81
	studentClipFPS    = 16
	studentClipWidth  = 832
	studentClipHeight = 480
	studentUpscale    = 4
)

// TestAFinishedClipIsEncodedOnNVENC: a finished clip is encoded on the card's NVENC as HEVC tagged hvc1 where the
// resolved FFmpeg lists it (the build OVERGO_FFMPEG names), the record saying so, and two writes of one clip are the
// same bytes, so a finished clip stays one clip. Measured on a Wan student clip of 81 frames at 832 x 480 finished at
// four times: the clip's whole write, the effect and the encode together. The effect alone took 3.8 s of it
// (rtx-vsr/nvenc, 16c3e5e8e), a timing no longer repeated (SQA-360).
func TestAFinishedClipIsEncodedOnNVENC(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_NVVFX") == "" {
		testskip.NotApplicable(t, "OVERGO_NVVFX names no NVIDIA Video Effects runtime here")
	}
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	path := filepath.Join(roots.Datasets, "VidGen", "self-forcing-generated", studentClip)
	if _, err := os.Stat(path); err != nil {
		testskip.NotApplicable(t, "the Wan student's VidGen clips are not on this machine: "+err.Error())
	}
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	pictures := testcheck.Must(media.DecodeVideoFile(t.Context(), path, ffmpeg, studentClipFPS, studentClipFrames))(t)
	frames := make([][]float32, len(pictures))
	for index, picture := range pictures {
		bounds := picture.Bounds()
		if bounds.Dx() != studentClipWidth || bounds.Dy() != studentClipHeight {
			t.Fatalf("frame %d of the student clip is %dx%d, want %dx%d", index, bounds.Dx(), bounds.Dy(), studentClipWidth, studentClipHeight)
		}
		plane := studentClipWidth * studentClipHeight
		frame := make([]float32, media.RGBChannels*plane)
		for y := range studentClipHeight {
			for x := range studentClipWidth {
				r, g, b, _ := picture.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
				for channel, value := range []uint32{r, g, b} {
					frame[channel*plane+y*studentClipWidth+x] = media.NormalizedRGBAChannel(value)
				}
			}
		}
		frames[index] = frame
	}
	if len(frames) != studentClipFrames {
		t.Fatalf("the student clip decoded to %d frames, want %d", len(frames), studentClipFrames)
	}
	feed := func(sink VideoFrameSink) error {
		for index, frame := range frames {
			if err := sink(index, frame, studentClipHeight, studentClipWidth); err != nil {
				return err
			}
		}
		return nil
	}
	finish := ClipFinish{Upscale: studentUpscale}
	var written []EncodedVideo
	var spent []time.Duration
	for range 2 {
		started := time.Now()
		written = append(written, testcheck.Must(WriteClip(t.Context(), mp4Clip, finish, studentClipFPS, UnitPixels, feed))(t))
		spent = append(spent, time.Since(started))
	}
	first := written[0]
	t.Logf("%d frames of %dx%d finished at %d times and encoded in %s and %s (%.1f frames a second), %d bytes, by %v",
		studentClipFrames, studentClipWidth, studentClipHeight, studentUpscale, spent[0], spent[1],
		float64(studentClipFrames)/spent[1].Seconds(), len(first.Data), first.Encoding.Args)
	if first.Width != studentUpscale*studentClipWidth || first.Height != studentUpscale*studentClipHeight || !media.OpensAsMP4(first.Data) {
		t.Fatalf("the finished clip is %dx%d, an MP4 %t", first.Width, first.Height, media.OpensAsMP4(first.Data))
	}
	// Spelled, not named, so this check reads the clip's own record of what wrote it.
	if !slices.Contains(first.Encoding.Args, "hevc_nvenc") || !slices.Contains(first.Encoding.Args, "hvc1") {
		t.Fatalf("the finished clip was written by %v, its finish %+v; want hevc_nvenc tagged hvc1", first.Encoding.Args, first.Encoding.Finish)
	}
	if !bytes.Equal(first.Data, written[1].Data) {
		t.Fatalf("two writes of one finished clip are %d and %d bytes and differ: NVENC does not repeat its bytes here", len(first.Data), len(written[1].Data))
	}
}
