package latentvideo

import "testing"

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
	for _, delay := range encoder.animation.Delay {
		total += delay
	}
	if total != 506 {
		t.Fatalf("81 frames at 16 fps encoded as %d centiseconds, want nearest centisecond to 506.25", total)
	}
}
