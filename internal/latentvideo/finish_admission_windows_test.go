//go:build windows

package latentvideo

import (
	"strings"
	"testing"

	"overgo/internal/media"
)

// TestAFinishWithoutItsRuntimeIsRefusedBeforeAnythingIsDrawn asks for an upscaled clip with the data root's record
// naming a directory that holds none of the effects' runtime, through a generation that records whether it ran
// (clip-encode/admit-runtime): the refusal names the command that records another runtime and comes before the
// generation, where Wan's 50 steps and its decode ran first and the refusal came at the first frame.
func TestAFinishWithoutItsRuntimeIsRefusedBeforeAnythingIsDrawn(t *testing.T) {
	recordedRuntimes(t, media.ClipRuntimes{Effects: t.TempDir()})
	const fps = 16
	drawn := false
	_, err := WriteClip(t.Context(), ClipFormat(""), ClipFinish{Upscale: 2}, fps, UnitPixels, func(VideoFrameSink) error {
		drawn = true
		return nil
	})
	if drawn {
		t.Errorf("the generation ran before a finish whose runtime is missing was refused (refusal: %v)", err)
	}
	if err == nil || !strings.Contains(err.Error(), "-clip-runtimes") {
		t.Errorf("a finish whose runtime is missing answered %v; want a refusal naming the command that records another", err)
	}
}
