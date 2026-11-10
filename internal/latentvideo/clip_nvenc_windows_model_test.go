//go:build windows && modeltest

package latentvideo

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

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

// studentClipFeed decodes the Wan student clip and feeds its frames, planar float32 RGB in [0, 1], to a clip's sink;
// it skips where the card, the effects runtime or the clip is absent.
func studentClipFeed(tb testing.TB) func(VideoFrameSink) error {
	tb.Helper()
	cudatest.Require(tb)
	if os.Getenv("OVERGO_NVVFX") == "" {
		testskip.NotApplicable(tb, "OVERGO_NVVFX names no NVIDIA Video Effects runtime here")
	}
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(tb)))(tb)
	path := filepath.Join(roots.Datasets, "VidGen", "self-forcing-generated", studentClip)
	if _, err := os.Stat(path); err != nil {
		testskip.NotApplicable(tb, "the Wan student's VidGen clips are not on this machine: "+err.Error())
	}
	ffmpeg := testcheck.Must(media.FFmpeg())(tb)
	pictures := testcheck.Must(media.DecodeVideoFile(tb.Context(), path, ffmpeg, studentClipFPS, studentClipFrames))(tb)
	frames := make([][]float32, len(pictures))
	for index, picture := range pictures {
		bounds := picture.Bounds()
		if bounds.Dx() != studentClipWidth || bounds.Dy() != studentClipHeight {
			tb.Fatalf("frame %d of the student clip is %dx%d, want %dx%d", index, bounds.Dx(), bounds.Dy(), studentClipWidth, studentClipHeight)
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
		tb.Fatalf("the student clip decoded to %d frames, want %d", len(frames), studentClipFrames)
	}
	return func(sink VideoFrameSink) error {
		for index, frame := range frames {
			if err := sink(index, frame, studentClipHeight, studentClipWidth); err != nil {
				return err
			}
		}
		return nil
	}
}

// TestAFinishedClipIsEncodedOnNVENC: a finished clip is encoded on the card's NVENC as HEVC tagged hvc1 where the
// resolved FFmpeg lists it (the build OVERGO_FFMPEG names), the record saying so, and two writes of one clip are the
// same bytes, so a finished clip stays one clip. Its time is BenchmarkFinishedClipWrite's.
func TestAFinishedClipIsEncodedOnNVENC(t *testing.T) {
	feed := studentClipFeed(t)
	finish := ClipFinish{Upscale: studentUpscale}
	var written []EncodedVideo
	for range 2 {
		written = append(written, testcheck.Must(WriteClip(t.Context(), mp4Clip, finish, studentClipFPS, UnitPixels, feed))(t))
	}
	first := written[0]
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

// BenchmarkFinishedClipWrite times a finished clip's whole write, the effect and the encode together, on the Wan
// student clip at two and four times, in frames a second; the effect alone took 3.8 s of the four-times write
// (rtx-vsr/nvenc, 16c3e5e8e). Opt in: go test -tags modeltest ./internal/latentvideo -run '^$' -bench
// '^BenchmarkFinishedClipWrite$' with OVERGO_CUDA_TEST=1 and OVERGO_NVVFX set.
func BenchmarkFinishedClipWrite(b *testing.B) {
	feed := studentClipFeed(b)
	for _, upscale := range []int{2, studentUpscale} {
		b.Run(fmt.Sprintf("x%d", upscale), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				testcheck.Must(WriteClip(b.Context(), mp4Clip, ClipFinish{Upscale: upscale}, studentClipFPS, UnitPixels, feed))(b)
			}
			b.ReportMetric(float64(studentClipFrames*b.N)/b.Elapsed().Seconds(), "frames/s")
		})
	}
}
