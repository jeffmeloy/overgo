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
