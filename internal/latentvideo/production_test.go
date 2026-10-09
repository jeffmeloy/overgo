package latentvideo

import (
	"bytes"
	"image/gif"
	"testing"

	"overgo/internal/media"
)

func TestVideoPixelRanges(t *testing.T) {
	tests := []struct {
		name   string
		pixels PixelRange
		input  []float32
		want   []byte
	}{
		{"signed", SignedUnitPixels, []float32{-1, 0, 1}, []byte{0, 128, 255}},
		{"unit", UnitPixels, []float32{-1, 0, 0.5, 1}, []byte{0, 0, 128, 255}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for index, value := range test.input {
				if got := encodeVideoByte(value, test.pixels); got != test.want[index] {
					t.Fatalf("encode %g=%d want=%d", value, got, test.want[index])
				}
			}
		})
	}
}

func TestGIFEncoderRejectsUnknownPixelRange(t *testing.T) {
	if _, err := NewGIFEncoder(16, PixelRange("unknown")); err == nil {
		t.Fatal("unknown pixel range accepted")
	}
}

func TestGIFEncoderFrameOrderAndTiming(t *testing.T) {
	if _, err := NewGIFEncoder(101, UnitPixels); err == nil {
		t.Fatal("unsupported GIF frame rate accepted")
	}
	encoder, err := NewGIFEncoder(16, UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	frame := []float32{0, 0, 0}
	if err := encoder.Add(1, frame, 1, 1); err == nil {
		t.Fatal("out-of-order first frame accepted")
	}
	for index := range 81 {
		if err := encoder.Add(index, frame, 1, 1); err != nil {
			t.Fatal(err)
		}
	}
	if err := encoder.Add(80, frame, 1, 1); err == nil {
		t.Fatal("duplicate frame index accepted")
	}
	total := 0
	for _, delay := range encoder.delays {
		total += delay
	}
	if total != 506 {
		t.Fatalf("81 frames at 16 fps encoded as %d centiseconds, want nearest centisecond to 506.25", total)
	}
}

// TestClipFramesDrawnTogetherAreTheSameClip holds a clip whose frames workers
// draw as they arrive to the clip of the same frames drawn one after another:
// the same bytes, in a palette cut from each frame and in the fixed one, with
// the count of changed pixels the frames' own bytes give.
func TestClipFramesDrawnTogetherAreTheSameClip(t *testing.T) {
	const frames, height, width, fps = 12, 40, 56, 16
	state := uint32(3)
	clip := make([][]float32, frames)
	for frame := range clip {
		clip[frame] = make([]float32, media.RGBChannels*height*width)
		for at := range clip[frame] {
			// A gradient that moves, under noise: each frame cuts another palette.
			state = state*1664525 + 1013904223
			clip[frame][at] = 0.8*float32((at%width+3*frame)%width)/float32(width) + 0.2*float32(state>>24)/float32(1<<8)
		}
	}
	for _, cut := range []bool{true, false} {
		encoder, err := NewGIFEncoder(fps, UnitPixels)
		if cut {
			encoder, err = NewClipEncoder(fps, UnitPixels)
		}
		if err != nil {
			t.Fatal(err)
		}
		var inTurn gif.GIF
		var prior []uint8
		changed := 0
		for index, frame := range clip {
			if err := encoder.Add(index, frame, height, width); err != nil {
				t.Fatal(err)
			}
			raw := make([]uint8, len(frame))
			if err := media.EncodePlanarRGB8Into(raw, frame, height, width, func(value float32) uint8 { return encodeVideoByte(value, UnitPixels) }); err != nil {
				t.Fatal(err)
			}
			for base := 0; prior != nil && base < len(raw); base += media.RGBChannels {
				if !bytes.Equal(raw[base:base+media.RGBChannels], prior[base:base+media.RGBChannels]) {
					changed++
				}
			}
			prior = raw
			delay, err := media.GIFFrameDelay(fps, index)
			if err != nil {
				t.Fatal(err)
			}
			inTurn.Image, inTurn.Delay = append(inTurn.Image, drawFrame(raw, height, width, cut)), append(inTurn.Delay, delay)
		}
		video, err := encoder.Finish()
		if err != nil {
			t.Fatal(err)
		}
		var want bytes.Buffer
		if err := gif.EncodeAll(&want, &inTurn); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(video.Data, want.Bytes()) || video.Frames != frames || video.ChangedPixels != changed || changed == 0 {
			t.Errorf("palette cut from each frame %t: the clip drawn by workers is %d bytes, %d frames and %d changed pixels; drawn in turn it is %d bytes, %d frames and %d changed pixels",
				cut, len(video.Data), video.Frames, video.ChangedPixels, want.Len(), frames, changed)
		}
	}
}
