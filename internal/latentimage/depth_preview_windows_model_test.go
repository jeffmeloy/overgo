//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/testcheck"
)

// TestTheDepthPreviewAgreesWithComfyUIs holds a depth section's preview (moss-ultra-compare/depth-preview) to the MOSS
// Ultra workflow's: the depth control given the recorder's picture at the recorded section's canvas makes the map
// ComfyUI's PreviewImage 667 showed of it, level for level, within depthPreviewTime (moss-speed/attention).
func TestTheDepthPreviewAgreesWithComfyUIs(t *testing.T) {
	recording := readComfySection(t, "pass-1", "depth-control")
	if recording.Set["513.image"] != "overgo-recorder-input.png" || recording.inputs("612:438")["resolution"] != float64(depthResolution) {
		t.Fatalf("the recording set %v with a detect resolution of %v; want the recorder's picture at %d", recording.Set, recording.inputs("612:438")["resolution"], depthResolution)
	}
	root := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "model_index.json"), []byte(`{"_class_name": "MOSSImagePipeline"}`), 0o600))
	ultraDepth(t, root)
	const width, height = 1024, 576
	asked := Request{Width: width, Height: height, Control: &DepthControl{Picture: base64.StdEncoding.EncodeToString(ultraPictureBytes(t))}}
	start := time.Now()
	preview := testcheck.Must(DepthPreview(root, asked))(t)
	if took := time.Since(start); took > depthPreviewTime {
		t.Errorf("the preview took %v; want it within %v", took, depthPreviewTime)
	} else {
		t.Logf("the preview took %v", took)
	}
	got := testcheck.Must(png.Decode(bytes.NewReader(preview.Data)))(t)
	want := testcheck.Must(png.Decode(bytes.NewReader(recording.picture(t, "667"))))(t)
	if got.Bounds() != want.Bounds() || got.Bounds() != image.Rect(0, 0, width, height) {
		t.Fatalf("the preview is %v and ComfyUI's %v; want the canvas, %d x %d", got.Bounds(), want.Bounds(), width, height)
	}
	equal, largest := 0, 0
	for y := range height {
		for x := range width {
			level := int(color.GrayModel.Convert(got.At(x, y)).(color.Gray).Y)
			r, g, b, _ := want.At(x, y).RGBA()
			if r != g || g != b {
				t.Fatalf("ComfyUI's preview at %d,%d is %d %d %d; want grey", x, y, r>>8, g>>8, b>>8)
			}
			difference := max(level-int(r>>8), int(r>>8)-level)
			largest = max(largest, difference)
			if difference == 0 {
				equal++
			}
		}
	}
	// Measured 2026-11-17: 585540 of 589824 levels equal, none more than one apart.
	if share := float64(equal) / float64(width*height); largest > 1 || share < leastEqualDepthShare {
		t.Errorf("the preview agrees with ComfyUI's at %d of %d pixels (%.4f), %d levels apart at most; want %.2f of them equal and none more than one apart",
			equal, width*height, share, largest, leastEqualDepthShare)
	}
}

// depthPreviewTime is the longest a depth preview of a 1024 x 576 picture takes on the host, its checkpoint loaded with
// it (moss-speed/attention): its estimate took 59 s with its encoder's attention on one thread, and 10.3 s with that
// attention's rows split across the host's workers beside a run of the cycle's lanes sharing the host, which this
// bound, twice that, holds on a busy host and the one thread fails.
const depthPreviewTime = 20 * time.Second

// leastEqualDepthShare is the share of a depth preview's levels that equal ComfyUI's at least: measured 0.9927, the
// rest one level apart where the host's float sums round otherwise than the card's.
const leastEqualDepthShare = 0.99
