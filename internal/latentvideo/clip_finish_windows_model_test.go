//go:build windows && modeltest

package latentvideo

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// finishReference is internal/cuda/nvvfx's reference frame, 320 x 180, value ((x (c + 1) + 2 y) mod 255) / 255, and
// the SHA-256 of NVIDIA's nvidia-vfx binding's output on it at QualityLevel 3 (high), upscaled by two, planar float32
// (scripts/python_golden_references/nvvfx_video_super_res.py; nvvfx TestVideoSuperResMatchesNvidiasBinding).
const (
	finishReferenceWidth  = 320
	finishReferenceHeight = 180
	finishReferencePeriod = 255
	finishBindingHigh     = "d9c0c6777af2263b8940414ff75aaac9ffa565ee66bd578ea119ae2128f585dd"
	finishClipFrames      = 6
	finishClipFPS         = 16
	runtimeLibraries      = 17
)

// captureEncoder keeps the frames a clip is handed, for a test to read.
type captureEncoder struct {
	frames        [][]float32
	height, width int
}

func (c *captureEncoder) add(_ context.Context, _ int, frame []float32, height, width int) error {
	c.frames, c.height, c.width = append(c.frames, append([]float32(nil), frame...)), height, width
	return nil
}

func (c *captureEncoder) Finish() (EncodedVideo, error) {
	return EncodedVideo{Frames: len(c.frames), Height: c.height, Width: c.width}, nil
}

func (c *captureEncoder) Close() error { return nil }

// TestAClipIsFinishedOnTheCardBeforeItIsEncoded: a clip asking for a finish has each frame run through NVIDIA's RTX
// Video Super Resolution on the card between its generator and its encoder. The frame the finish hands on is NVIDIA's
// binding's own, to the bit, at the level and scale asked; a clip written through WriteClip as an MP4, denoised and
// upscaled by two, is written at twice its frames' size, and its encoding names the levels run in order (denoise
// before upscale, as NVIDIA's script runs them), the scale, and the runtime's seventeen libraries by their digests.
func TestAClipIsFinishedOnTheCardBeforeItIsEncoded(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_NVVFX") == "" {
		testskip.NotApplicable(t, "OVERGO_NVVFX names no NVIDIA Video Effects runtime here")
	}
	frame := make([]float32, media.RGBChannels*finishReferenceHeight*finishReferenceWidth)
	plane := finishReferenceHeight * finishReferenceWidth
	for channel := range media.RGBChannels {
		for y := range finishReferenceHeight {
			for x := range finishReferenceWidth {
				frame[channel*plane+y*finishReferenceWidth+x] = float32((x*(channel+1)+2*y)%finishReferencePeriod) / finishReferencePeriod
			}
		}
	}
	capture := &captureEncoder{}
	finishing := testcheck.Must(newFinishingEncoder(capture, ClipFinish{Upscale: 2, UpscaleLevel: "high"}, UnitPixels))(t)
	testcheck.NoError(t, finishing.add(t.Context(), 0, frame, finishReferenceHeight, finishReferenceWidth))
	testcheck.NoError(t, finishing.release())
	written := make([]byte, len(capture.frames[0])*float32Bytes)
	for index, value := range capture.frames[0] {
		binary.LittleEndian.PutUint32(written[index*float32Bytes:], math.Float32bits(value))
	}
	digest := sha256.Sum256(written)
	if got := hex.EncodeToString(digest[:]); capture.width != 2*finishReferenceWidth || capture.height != 2*finishReferenceHeight || got != finishBindingHigh {
		t.Fatalf("the finish handed on a %dx%d frame hashing %s, want NVIDIA's binding's %dx%d frame %s", capture.width, capture.height, got,
			2*finishReferenceWidth, 2*finishReferenceHeight, finishBindingHigh)
	}

	signed := make([]float32, len(frame))
	for index, value := range frame {
		signed[index] = 2*value - 1
	}
	finish := ClipFinish{Upscale: 2, Denoise: 1}
	started := time.Now()
	video := testcheck.Must(WriteClip(t.Context(), mp4Clip, finish, finishClipFPS, SignedUnitPixels, func(sink VideoFrameSink) error {
		for index := range finishClipFrames {
			if err := sink(index, signed, finishReferenceHeight, finishReferenceWidth); err != nil {
				return err
			}
		}
		return nil
	}))(t)
	spent := time.Since(started)
	t.Logf("%d frames of %dx%d denoised and upscaled to %dx%d and written as MP4 in %s (%.1f frames a second)", finishClipFrames,
		finishReferenceWidth, finishReferenceHeight, video.Width, video.Height, spent, float64(finishClipFrames)/spent.Seconds())
	record := video.Encoding.Finish
	if !media.OpensAsMP4(video.Data) || video.Width != 2*finishReferenceWidth || video.Height != 2*finishReferenceHeight || record == nil {
		t.Fatalf("the finished clip is %dx%d, an MP4 %t, its finish %+v", video.Width, video.Height, media.OpensAsMP4(video.Data), record)
	}
	if len(record.Levels) != 2 || record.Levels[0] != denoiseLevelBase+1 || record.Levels[1] != upscaleLevels[defaultUpscaleLevel] ||
		record.Scale != 2 || len(record.Runtime) != runtimeLibraries {
		t.Fatalf("the finish records levels %v at scale %d, %d runtime libraries; want [%d %d] at 2 and %d", record.Levels, record.Scale,
			len(record.Runtime), denoiseLevelBase+1, upscaleLevels[defaultUpscaleLevel], runtimeLibraries)
	}
}
