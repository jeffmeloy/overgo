package trainingdata

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/testutil"
)

func TestAudioProcessorDecodesTypedWAV(t *testing.T) {
	t.Parallel()
	want := []float32{0.5, -0.5}
	example, err := AudioProcessor(RoleInput)(context.Background(), RawRecord{
		ID: "audio", Group: "audio", Data: testutil.MonoPCM16WAV(16000, []int16{16384, -16384}),
	})
	if err != nil || len(example.Values) != 1 {
		t.Fatalf("audio example=%+v: %v", example, err)
	}
	got, rate, err := Audio(example.Values[0])
	if err != nil || rate != 16000 || !slices.Equal(got, want) {
		t.Fatalf("audio=%v/%d, want %v/16000: %v", got, rate, want, err)
	}
}
