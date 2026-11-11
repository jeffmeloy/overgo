//go:build windows

package latentvideo

import (
	"strings"
	"testing"
)

// TestAFinishWithoutItsRuntimeIsRefusedBeforeAnythingIsDrawn asks for an upscaled clip with OVERGO_NVVFX naming a
// directory that holds none of the effects' runtime, through a generation that records whether it ran
// (clip-encode/admit-runtime): the refusal names the setting and comes before the generation, where Wan's 50 steps and
// its decode ran first and the refusal came at the first frame.
func TestAFinishWithoutItsRuntimeIsRefusedBeforeAnythingIsDrawn(t *testing.T) {
	t.Setenv("OVERGO_NVVFX", t.TempDir())
	const fps = 16
	drawn := false
	_, err := WriteClip(t.Context(), ClipFormat(""), ClipFinish{Upscale: 2}, fps, UnitPixels, func(VideoFrameSink) error {
		drawn = true
		return nil
	})
	if drawn {
		t.Errorf("the generation ran before a finish whose runtime is missing was refused (refusal: %v)", err)
	}
	if err == nil || !strings.Contains(err.Error(), "OVERGO_NVVFX") {
		t.Errorf("a finish whose runtime is missing answered %v; want a refusal naming OVERGO_NVVFX", err)
	}
}
