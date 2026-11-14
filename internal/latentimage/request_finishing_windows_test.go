//go:build windows

package latentimage

import (
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// TestARequestsCanvasAndSharpeningFinishItsPicture holds the MOSS Ultra workflow's finishing on a request
// (moss-ultra-sections/finishing): a canvas of 16:9 at 0.5625 megapixels in multiples of 32 sizes the picture 1024 × 576
// as the workflow's recordings were drawn, a canvas beside a width and height is refused, and a sharpening is asked of a
// rotary profile alone and held to its bounds.
func TestARequestsCanvasAndSharpeningFinishItsPicture(t *testing.T) {
	canvas := &media.Canvas{Aspect: "16:9", Megapixels: 0.5625, Multiple: 32}
	sized := testcheck.Must(Request{Prompt: "a lighthouse", Canvas: canvas}.withPreset(""))(t)
	if sized.Width != 1024 || sized.Height != 576 || sized.Canvas != nil {
		t.Errorf("the canvas sized %d×%d (canvas left %v), want the recordings' 1024×576", sized.Width, sized.Height, sized.Canvas)
	}
	if _, err := (Request{Prompt: "a lighthouse", Width: 512, Height: 512, Canvas: canvas}).withPreset(""); err == nil {
		t.Error("a canvas beside a width and height was taken")
	}
	sharpen := &media.Sharpening{Radius: 1, Sigma: 0.6, Alpha: 0.7}
	if !(Request{Sharpen: sharpen}).rotaryOnly() {
		t.Error("a sharpening was offered to a flow profile")
	}
	testcheck.NoError(t, ValidateRequest(Request{Prompt: "a lighthouse", Width: 1024, Height: 576, Sharpen: sharpen}))
	if ValidateRequest(Request{Prompt: "a lighthouse", Width: 1024, Height: 576, Sharpen: &media.Sharpening{Radius: 0, Sigma: 0.6}}) == nil {
		t.Error("a sharpening of radius 0 was taken")
	}
	// A radius as wide as the picture is refused before the model opens, not after it draws (sharpen-admit/do, SQA-393).
	if ValidateRequest(Request{Prompt: "a lighthouse", Width: 1024, Height: 576, Sharpen: &media.Sharpening{Radius: 576, Sigma: 0.6, Alpha: 0.7}}) == nil {
		t.Error("a sharpening whose radius the picture cannot hold was taken")
	}
}
