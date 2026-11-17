//go:build modeltest

package vitencoder

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// The depth section as the MOSS Ultra workflow runs it at 1024 x 576 (comfy-recorder/record-rest, pass-1): the
// recorder's picture scaled to the canvas (612:514, lanczos, centre crop), the preprocessor at its resolution of 512
// (612:438), the map scaled back to the canvas (612:439, bilinear, no crop), and the preview of it ComfyUI kept (667).
const (
	depthRecording  = "fixtures/comfy/moss-ultra/recordings/pass-1/depth-control.json"
	depthPreview    = "667"
	depthPicture    = "8f6948d0518656d9d19c2dc797e05e95fd990e37d84eadf0436d52be9de53d8a"
	depthResolution = 512
	depthWidth      = 1024
	depthHeight     = 576
)

// TestDepthMapIsWithinOneLevelOfTheRecordedOne holds overgo's Depth Anything V2 Small (moss-ultra-sections/depth) to
// the MOSS Ultra workflow's depth map as ComfyUI made it: the checkpoint the workflow loads, read from its PyTorch file
// through internal/vitencoder's DINOv2 encoder and the DPT head, given the section's picture and run as
// comfyui_controlnet_aux's DepthAnythingV2Preprocessor runs it, then scaled to the canvas as ComfyUI's ImageScale scales
// it and kept as PreviewImage keeps a picture, gives every pixel of ComfyUI's preview within one 8-bit level.
func TestDepthMapIsWithinOneLevelOfTheRecordedOne(t *testing.T) {
	repo := testutil.RepoRoot(t)
	references := filepath.Join(testcheck.Must(dataroot.Resolve(repo))(t).References(), "comfy", "moss-ultra")
	var recording struct {
		Set   map[string]any `json:"set"`
		Files []struct {
			Node   string `json:"node"`
			Name   string `json:"name"`
			SHA256 string `json:"sha256"`
		} `json:"files"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, filepath.FromSlash(depthRecording))))(t), &recording))
	if recording.Set["513.image"] != "overgo-recorder-input.png" || recording.Set["690.megapixels"] != 0.5625 {
		t.Fatalf("the recording set %v; want the recorder's picture at 1024 x 576", recording.Set)
	}
	read := func(path, digest string) []byte {
		t.Helper()
		data := testcheck.Must(os.ReadFile(path))(t)
		if sum := sha256.Sum256(data); hex.EncodeToString(sum[:]) != digest {
			t.Fatalf("%s hashes %x; want %s", path, sum, digest)
		}
		return data
	}
	var preview image.Image
	for _, file := range recording.Files {
		if file.Node == depthPreview {
			data := read(filepath.Join(references, "pass-1", "depth-control", file.Name), file.SHA256)
			preview = testcheck.Must(png.Decode(bytes.NewReader(data)))(t)
		}
	}
	if preview == nil || preview.Bounds().Dx() != depthWidth || preview.Bounds().Dy() != depthHeight {
		t.Fatalf("the recording keeps no %d x %d preview of the depth map", depthWidth, depthHeight)
	}
	picture := testcheck.Must(png.Decode(bytes.NewReader(read(filepath.Join(references, "overgo-recorder-input.png"), depthPicture))))(t)

	// The section's picture as ComfyUI holds it on the canvas.
	levels := testcheck.Must(media.ScaleLikeComfyUI(picture, depthWidth, depthHeight))(t)
	unit := make([]float32, len(levels))
	for index, level := range levels {
		unit[index] = float32(level) / media.UnitLevels
	}
	model := testcheck.Must(LoadDepthAnything(filepath.Join(testutil.ModelArtifactDir(t, "depth-anything-v2-small"), "depth_anything_v2_vits.pth")))(t)
	detected, mapHeight, mapWidth := func() ([]uint8, int, int) {
		detected, mapHeight, mapWidth, err := model.DepthMap(unit, depthHeight, depthWidth, depthResolution)
		testcheck.NoError(t, err)
		return detected, mapHeight, mapWidth
	}()
	if mapHeight != 512 || mapWidth != 910 {
		t.Fatalf("the map is %d x %d; want the preprocessor's 910 x 512", mapWidth, mapHeight)
	}
	shown := make([]float32, len(detected))
	for index, level := range detected {
		shown[index] = float32(level) / media.UnitLevels
	}
	canvas := testcheck.Must(media.BilinearLikeComfyUI(shown, mapHeight, mapWidth, 1, depthHeight, depthWidth))(t)
	var counts [256]int
	worst := 0
	for y := range depthHeight {
		for x := range depthWidth {
			got := int(uint8(min(max(canvas[y*depthWidth+x]*255, 0), 255)))
			r, g, b, _ := preview.At(x, y).RGBA()
			want := int(r >> 8)
			if int(g>>8) != want || int(b>>8) != want {
				t.Fatalf("the preview's pixel (%d, %d) is not grey", x, y)
			}
			difference := max(got-want, want-got)
			counts[difference]++
			worst = max(worst, difference)
		}
	}
	t.Logf("of %d pixels: %d equal, %d one level off, %d more; the worst %d levels", depthWidth*depthHeight, counts[0], counts[1], depthWidth*depthHeight-counts[0]-counts[1], worst)
	if worst > 1 {
		t.Errorf("the depth map lies up to %d levels from ComfyUI's; want within one", worst)
	}
}
