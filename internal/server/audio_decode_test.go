package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"slices"
	"testing"

	"overgo/internal/testutil"
)

func TestAudioDecodeConsolidationAcceptance(t *testing.T) {
	wav := testutil.MonoPCM16WAV(16000, []int16{16384, -16384})
	encoded := base64.StdEncoding.EncodeToString(wav)
	handler := &Handler{}
	for name, source := range map[string]string{"base64": encoded, "data-uri": "data:audio/wav;base64," + encoded} {
		t.Run(name, func(t *testing.T) {
			got, size, err := handler.resolveAudioData(t.Context(), source, "wav")
			if err != nil || size < len(wav) || !slices.Equal(got, []float32{.5, -.5}) {
				t.Fatalf("samples=%v bytes=%d error=%v", got, size, err)
			}
			ctx, cancel := context.WithCancelCause(t.Context())
			cancel(context.Canceled)
			got, _, err = handler.resolveAudioData(ctx, source, "wav")
			if !errors.Is(err, context.Canceled) || got != nil {
				t.Fatalf("cancelled samples=%v error=%v", got, err)
			}
		})
	}
	stereo := slices.Clone(wav)
	binary.LittleEndian.PutUint16(stereo[22:], 2)
	binary.LittleEndian.PutUint32(stereo[28:], 64000)
	binary.LittleEndian.PutUint16(stereo[32:], 4)
	for name, data := range map[string][]byte{
		"stereo":         stereo,
		"wrong-rate":     testutil.MonoPCM16WAV(8000, []int16{16384}),
		"truncated":      wav[:len(wav)-1],
		"disguised-flac": []byte("fLaC"),
	} {
		t.Run(name, func(t *testing.T) {
			got, _, err := handler.resolveAudioData(t.Context(), base64.StdEncoding.EncodeToString(data), "wav")
			if err == nil || got != nil {
				t.Fatalf("refused input samples=%v error=%v", got, err)
			}
		})
	}
	t.Run("wire-format-refused", func(t *testing.T) {
		if got, _, err := handler.resolveAudioData(t.Context(), encoded, "flac"); err == nil || got != nil {
			t.Fatalf("unsupported format samples=%v error=%v", got, err)
		}
	})
}
