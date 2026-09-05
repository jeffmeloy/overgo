package main

import (
	"context"
	"encoding/binary"
	"errors"
	"slices"
	"testing"

	"overgo/internal/binaryschema"
	"overgo/internal/testutil"
)

func TestAudioDecodeConsolidationAcceptance(t *testing.T) {
	want := []float32{.5, -.5}
	wav := testutil.MonoPCM16WAV(16000, []int16{16384, -16384})
	for extension, data := range map[string][]byte{".wav": wav, ".f32": binaryschema.LittleEndian.Float32s(want)} {
		t.Run(extension, func(t *testing.T) {
			got, err := decodeProjectedAudio(t.Context(), data, extension, 16000)
			if err != nil || !slices.Equal(got, want) {
				t.Fatalf("samples=%v error=%v", got, err)
			}
			ctx, cancel := context.WithCancelCause(t.Context())
			cancel(context.Canceled)
			got, err = decodeProjectedAudio(ctx, data, extension, 16000)
			if !errors.Is(err, context.Canceled) || got != nil {
				t.Fatalf("cancelled samples=%v error=%v", got, err)
			}
		})
	}
	stereo := slices.Clone(wav)
	binary.LittleEndian.PutUint16(stereo[22:], 2)
	binary.LittleEndian.PutUint32(stereo[28:], 64000)
	binary.LittleEndian.PutUint16(stereo[32:], 4)
	for _, test := range []struct {
		name, extension string
		data            []byte
		rate            int
	}{
		{"stereo", ".wav", stereo, 16000},
		{"wrong-rate", ".wav", wav, 8000},
		{"truncated", ".wav", wav[:len(wav)-1], 16000},
		{"disguised-flac", ".wav", []byte("fLaC"), 16000},
		{"unsupported-extension", ".flac", wav, 16000},
		{"partial-float", ".f32", []byte{0}, 16000},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := decodeProjectedAudio(t.Context(), test.data, test.extension, test.rate)
			if err == nil || got != nil {
				t.Fatalf("refused input samples=%v error=%v", got, err)
			}
		})
	}
}
