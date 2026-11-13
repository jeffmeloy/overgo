//go:build windows && modeltest

package latentvideo

import (
	"bytes"
	"context"
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// comparedFrames are the frames of each finished clip decoded and compared: the first of the clip, eight at
// 3328 x 1920, which an image of each holds in about 200 MB.
const comparedFrames = 8

// floatOnly hides the YUV path of the encoder it wraps, so a finish hands it float32 RGB frames as it did before
// clip-encode/device-bytes, and keeps the first comparedFrames of them, the finished frames both clips are read against.
type floatOnly struct {
	clipEncoder
	kept [][]float32
}

func (f *floatOnly) add(ctx context.Context, index int, frame []float32, height, width int) error {
	if len(f.kept) < comparedFrames {
		f.kept = append(f.kept, slices.Clone(frame))
	}
	return f.clipEncoder.add(ctx, index, frame, height, width)
}

// TestAFinishedFrameReachesTheEncoderAsEightBitYUV finishes the Wan student clip at four times (clip-encode/device-bytes):
// each finished frame is converted on the card to 8-bit BT.709 limited-range YUV 4:2:0 and read back as the bytes FFmpeg
// encodes, width x height x 3/2 a frame and no float32 copy kept; two writes are the same bytes; and against the finished
// frames, each of the clip's frames reads no more than half a decibel under the clip written the old way, read back as
// float32 RGB and converted on the host. Measured 2026-11-13, building: with the raw YUV untagged, FFmpeg read it as
// BT.601 and converted it to the BT.709 it was told to write, a uniform 3.2 levels dark, 35 dB against the old way's 40.
func TestAFinishedFrameReachesTheEncoderAsEightBitYUV(t *testing.T) {
	feed := studentClipFeed(t)
	finish := ClipFinish{Upscale: studentUpscale}
	// A finished frame's YUV 4:2:0: a luma byte a pixel, and a U and a V byte for each 2 x 2 block.
	frameBytes := studentUpscale * studentClipWidth * studentUpscale * studentClipHeight * 3 / 2
	var written []EncodedVideo
	for range 2 {
		encoder := testcheck.Must(newServedEncoder(t.Context(), mp4Clip, finish, studentClipFPS, UnitPixels))(t)
		finishing, finished := encoder.(*finishingEncoder)
		video := testcheck.Must(writeClip(t.Context(), encoder, feed))(t)
		if !finished || len(finishing.result) != frameBytes || finishing.finished != nil {
			t.Fatalf("a finished frame was read back as %d bytes, a float32 copy kept %t; want the %d bytes of its YUV 4:2:0 and none",
				len(finishing.result), finishing.finished != nil, frameBytes)
		}
		written = append(written, video)
	}
	if !bytes.Equal(written[0].Data, written[1].Data) {
		t.Fatalf("two writes of one finished clip are %d and %d bytes and differ", len(written[0].Data), len(written[1].Data))
	}
	next, encodedBy, err := newFormatEncoder(t.Context(), mp4Clip, studentClipFPS, UnitPixels, true)
	testcheck.NoError(t, err)
	floats := &floatOnly{clipEncoder: next}
	finishing := testcheck.Must(newFinishingEncoder(floats, finish, UnitPixels))(t)
	finishing.encodedBy = encodedBy
	old := testcheck.Must(writeClip(t.Context(), finishing, feed))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	decode := func(video EncodedVideo, name string) []image.Image {
		t.Helper()
		path := filepath.Join(t.TempDir(), name)
		testcheck.NoError(t, os.WriteFile(path, video.Data, 0o600))
		return testcheck.Must(media.DecodeVideoFile(t.Context(), path, ffmpeg, studentClipFPS, comparedFrames))(t)
	}
	now, before := decode(written[0], "yuv.mp4"), decode(old, "float.mp4")
	// Each clip is lossy, so the two are read against the frames they encode, the effect's own output, and the clip
	// written in YUV is held to no more than half a decibel under the one written the old way: read against each other
	// the two encodes differ by both their losses, about 37 dB.
	for index := range comparedFrames {
		yuv, float := framePSNR(now[index], floats.kept[index]), framePSNR(before[index], floats.kept[index])
		if yuv < float-0.5 {
			t.Errorf("frame %d of the clip written in YUV reads %.2f dB against the finished frame, the one written in float32 %.2f", index, yuv, float)
		} else {
			t.Logf("frame %d against the finished frame: %.2f dB written in YUV, %.2f written in float32", index, yuv, float)
		}
	}
}

// framePSNR is the peak signal-to-noise ratio of a decoded frame's 8-bit RGB against a finished frame, planar float32
// RGB in [0, 1] of the same size, in decibels.
func framePSNR(decoded image.Image, finished []float32) float64 {
	bounds := decoded.Bounds()
	plane := bounds.Dx() * bounds.Dy()
	var squared float64
	for y := range bounds.Dy() {
		for x := range bounds.Dx() {
			r, g, b, _ := decoded.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			for channel, value := range []uint32{r, g, b} {
				difference := float64(value>>8) - math.Round(float64(finished[channel*plane+y*bounds.Dx()+x])*255)
				squared += difference * difference
			}
		}
	}
	mean := squared / float64(media.RGBChannels*plane)
	if mean == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(255*255/mean)
}
