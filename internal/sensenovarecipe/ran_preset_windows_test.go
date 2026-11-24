//go:build windows

package sensenovarecipe

import (
	"testing"

	"overgo/internal/media"
)

// TestASenseNovaRequestRecordsWhatItRan holds image-presets for SenseNova: a validated request names the preset its
// picture records, its size, its steps of flow Euler, its guidance scale and its timestep shift, so the scorecard shows
// its served time with what it is a time of; the full measurement of 2026-11-24 left SenseNova's first output unshown,
// "its measured attempt recorded no preset". The decoded picture carries it (TestFrozenSenseNovaGeneration).
func TestASenseNovaRequestRecordsWhatItRan(t *testing.T) {
	t.Parallel()
	request := GenerationRequest{Prompt: "an infographic", Width: 512, Height: 384, Steps: 30, CFGScale: 4, TimestepShift: 3}
	want := media.Preset{Width: 512, Height: 384, Steps: 30, Guidance: 4, Sampler: "flow Euler", Shift: 3}
	if got := request.ran(); got != want || !got.Ran() {
		t.Fatalf("the request records the preset %+v; want %+v, one that ran", got, want)
	}
}
