package trainingdata

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"slices"
	"testing"

	"overgo/internal/testutil"
)

func TestAudioDecodeConsolidationAcceptance(t *testing.T) {
	t.Parallel()
	want := []float32{0.5, -0.5}
	wav := testutil.MonoPCM16WAV(16000, []int16{16384, -16384})
	// libsndfile 1.2.2 / libFLAC 1.4.3 encodes and independently decodes
	// these two mono PCM16 samples as [.5, -.5] at 16 kHz.
	flac, err := hex.DecodeString("664c6143000000221000100000000d00000d03e800f000000002097732d2b267dfd70590c83568c6a10484000028200000007265666572656e6365206c6962464c414320312e342e3320323032333036323300000000fff865080001f2030005c07968")
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{"wav": wav, "flac": flac} {
		t.Run(name, func(t *testing.T) {
			example, err := AudioProcessor(RoleInput, uint64(len(want)))(t.Context(), RawRecord{ID: "audio", Group: "source", Data: data})
			if err != nil || len(example.Values) != 1 || example.ID != "audio" || example.Group != "source" || example.Values[0].Role != RoleInput {
				t.Fatalf("audio example=%+v: %v", example, err)
			}
			got, rate, err := Audio(example.Values[0])
			if err != nil || rate != 16000 || !slices.Equal(got, want) {
				t.Fatalf("audio=%v/%d, want %v/16000: %v", got, rate, want, err)
			}
		})
	}
	stereo := slices.Clone(wav)
	binary.LittleEndian.PutUint16(stereo[22:], 2)
	binary.LittleEndian.PutUint32(stereo[28:], 64000)
	binary.LittleEndian.PutUint16(stereo[32:], 4)
	for _, test := range []struct {
		name  string
		data  []byte
		limit uint64
	}{
		{"wav-budget", wav, 1}, {"flac-budget", flac, 1}, {"zero-budget", wav, 0},
		{"stereo", stereo, 2}, {"truncated", wav[:len(wav)-1], 2}, {"unsupported", []byte("OggS"), 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			example, err := AudioProcessor(RoleInput, test.limit)(t.Context(), RawRecord{Data: test.data})
			if err == nil || len(example.Values) != 0 {
				t.Fatalf("refused input produced %+v: %v", example, err)
			}
		})
	}
	t.Run("cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		example, err := AudioProcessor(RoleInput, 2)(ctx, RawRecord{Data: wav})
		if !errors.Is(err, context.Canceled) || len(example.Values) != 0 {
			t.Fatalf("cancelled input produced %+v: %v", example, err)
		}
	})
}
