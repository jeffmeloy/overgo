package media

import (
	"context"
	"encoding/hex"
	"errors"
	"math"
	"slices"
	"testing"

	"overgo/internal/binaryschema"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestAudioDecodeConsolidationAcceptance(t *testing.T) {
	want := []float32{0.5, -0.5}
	encoded := binaryschema.LittleEndian.Float32s(want)
	decoded, err := DecodeFloat32LE(encoded)
	if err != nil || !slices.Equal(decoded, want) {
		t.Fatalf("decoded = %v, error = %v", decoded, err)
	}
	wav := testutil.MonoPCM16WAV(16000, []int16{16384, -16384})
	wave, status, err := DecodeAudio(t.Context(), wav, uint64(len(want)))
	if err != nil || status != recipecontract.AudioDecodeComplete || wave.Format.SampleRate != 16000 || wave.Format.Channels != 1 || !slices.Equal(wave.Samples, want) {
		t.Fatalf("wave = %+v, status=%s error=%v", wave, status, err)
	}
	t.Run("provisional-output", func(t *testing.T) {
		var delivered []float32
		cause := errors.New("output closed")
		ctx := WithAudioOutput(t.Context(), func(_ context.Context, chunk DecodedAudio) error { delivered = chunk.Samples; return cause })
		if err := EmitAudio(ctx, wave); !errors.Is(err, cause) || !slices.Equal(delivered, want) {
			t.Fatalf("consumer: samples=%v err=%v", delivered, err)
		}
		for _, samples := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(1))}} {
			delivered = nil
			invalid := wave
			invalid.Samples = samples
			if err := EmitAudio(ctx, invalid); err == nil || delivered != nil {
				t.Fatalf("invalid output reached consumer: %v %v", samples, err)
			}
		}
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		if err := EmitAudio(cancelled, wave); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled output: %v", err)
		}
	})
	flac, err := hex.DecodeString(stereoFLACHex)
	if err != nil {
		t.Fatal(err)
	}
	t.Run("flac-preserves-channels", func(t *testing.T) {
		audio, status, err := DecodeAudio(t.Context(), flac, 48)
		if err != nil || status != recipecontract.AudioDecodeComplete || audio.Format.Channels != 2 || len(audio.Samples) != 48 || !slices.Equal(audio.Samples[:6], []float32{0, .5, -.5, .25, .125, -.125}) {
			t.Fatalf("audio=%+v status=%s error=%v", audio, status, err)
		}
	})
	for _, test := range []struct {
		name   string
		data   []byte
		limit  uint64
		status recipecontract.AudioDecodeStatus
	}{
		{"wav-budget", wav, 1, recipecontract.AudioDecodeResourceLimit},
		{"flac-budget", flac, 47, recipecontract.AudioDecodeResourceLimit},
		{"unsupported", []byte("OggS"), 1, recipecontract.AudioDecodeUnsupported},
		{"truncated", wav[:len(wav)-1], 2, recipecontract.AudioDecodeTruncated},
	} {
		t.Run(test.name, func(t *testing.T) {
			audio, status, err := DecodeAudio(t.Context(), test.data, test.limit)
			if err == nil || status != test.status || audio.Samples != nil {
				t.Fatalf("audio=%+v status=%s error=%v", audio, status, err)
			}
		})
	}
	t.Run("cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		audio, _, err := DecodeAudio(ctx, wav, 2)
		if !errors.Is(err, context.Canceled) || audio.Samples != nil {
			t.Fatalf("audio=%+v error=%v", audio, err)
		}
	})
}
