package media

import (
	"slices"
	"testing"

	"llamacpp2go/internal/testutil"
)

func TestAudioCodecs(t *testing.T) {
	want := []float32{0.5, -0.5}
	encoded := EncodeFloat32LE(want)
	decoded, err := DecodeFloat32LE(encoded)
	if err != nil || !slices.Equal(decoded, want) {
		t.Fatalf("decoded = %v, error = %v", decoded, err)
	}
	wave, rate, err := DecodeWAV(testutil.MonoPCM16WAV(16000, []int16{16384, -16384}))
	if err != nil || rate != 16000 || !slices.Equal(wave, want) {
		t.Fatalf("wave = %v/%d, error = %v", wave, rate, err)
	}
}
