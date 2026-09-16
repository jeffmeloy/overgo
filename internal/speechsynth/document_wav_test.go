package speechsynth

import (
	"bytes"
	"overgo/internal/media"
	"testing"
)

func TestWAVSegmentOrder(t *testing.T) {
	wave, err := encodeDocumentWAV([][]float32{{0, 0.5}, nil, {-0.5, 2}, {-2}}, 24000)
	if err != nil {
		t.Fatal(err)
	}
	// Independent PCM16 little-endian expectations include clipping across
	// segment boundaries and an empty intermediate span.
	want := []byte{0, 0, 0, 64, 0, 192, 255, 127, 0, 128}
	if !bytes.Equal(wave[len(wave)-len(want):], want) {
		t.Fatal("segment order or PCM encoding changed")
	}
	audio, status, err := media.DecodeAudio(t.Context(), wave, 5)
	if err != nil || len(audio.Samples) != 5 || audio.Format.SampleRate != 24000 {
		t.Fatalf("invalid joined WAV: %s %+v %v", status, audio, err)
	}
}
