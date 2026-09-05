package latentvideo

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/media"
)

// TestEditPolicyDeclaresTheReferenceSchedule pins the declared LiveEdit
// schedule against the upstream few-step configuration, resolved from a
// directory carrying the checkpoint and refused from one without it.
func TestEditPolicyDeclaresTheReferenceSchedule(t *testing.T) {
	models := t.TempDir()
	edit := filepath.Join(models, "LiveEdit")
	if err := os.MkdirAll(edit, 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := ResolveEditPolicy(edit); err == nil {
		t.Fatal("a directory without the checkpoint resolved a policy")
	}
	wanDirectory, checkpoint := LiveEditArtifacts(edit)
	if wanDirectory != filepath.Join(models, "Wan2.1-T2V-1.3B") || checkpoint != filepath.Join(edit, "ar-forcing_002000.pt") {
		t.Fatalf("artifacts = %s, %s", wanDirectory, checkpoint)
	}
	if err := os.WriteFile(checkpoint, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	policy, err := ResolveEditPolicy(edit)
	if err != nil {
		t.Fatal(err)
	}
	if policy.FramesPerChunk != 3 || policy.LocalAttentionFrames != 21 || policy.InferenceSteps != 1000 ||
		policy.Shift != 5 || policy.SigmaMin != 0 || policy.SigmaMax != 1 || !policy.ExtraStep || policy.ContextTimestep != 0 ||
		len(policy.DenoisingTimesteps) != 4 || policy.DenoisingTimesteps[0] != 1000 || policy.DenoisingTimesteps[3] != 250 {
		t.Fatalf("policy = %+v", policy)
	}
	// The denoiser's chunk arithmetic: whole chunks beyond the first, inside
	// the local attention history.
	for latent, admitted := range map[int]bool{3: false, 4: false, 6: true, 9: true, 24: true, 27: false} {
		if err := policy.admitLatentFrames(latent); (err == nil) != admitted {
			t.Errorf("latent frames %d admitted=%t err=%v", latent, err == nil, err)
		}
	}
}

// TestReferenceEditRequestFormAdmitsClipOrTensors pins the two request
// forms: the clip form names a prompt, a seed and a source artifact and
// marshals without condition or source keys, and the tensor form keeps
// its strict validation.
func TestReferenceEditRequestFormAdmitsClipOrTensors(t *testing.T) {
	clip := ReferenceEditRequest{Prompt: "a fox", Seed: 7, SourceArtifact: "file:sha256:" + strings.Repeat("ab", 32)}
	if !clip.ClipForm() {
		t.Fatal("clip form not recognised")
	}
	if err := ValidateReferenceEditRequestForm(clip); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(clip)
	if err != nil {
		t.Fatal(err)
	}
	for _, absent := range []string{`"condition"`, `"source"`} {
		if strings.Contains(string(encoded), absent) {
			t.Errorf("clip form carries %s: %s", absent, encoded)
		}
	}
	for name, request := range map[string]ReferenceEditRequest{
		"empty":            {},
		"prompt only":      {Prompt: "a fox"},
		"artifact only":    {SourceArtifact: clip.SourceArtifact},
		"unparseable clip": {Prompt: "a fox", SourceArtifact: "not-an-id"},
		"negative seed":    {Prompt: "a fox", Seed: -1, SourceArtifact: clip.SourceArtifact},
	} {
		if ValidateReferenceEditRequestForm(request) == nil {
			t.Errorf("%s: admitted %+v", name, request)
		}
	}
	tensor := ReferenceEditRequest{
		Condition: ReferenceEditCondition{
			TextContext: []float32{1}, FramesPerChunk: 1, LocalAttention: 1, Timesteps: []int64{900}, Sigmas: []float32{0.9},
		},
		Source: SourceVideo{Pixels: make([]float32, 3*1*2*2), Channels: 3, Frames: 1, Height: 2, Width: 2},
	}
	if tensor.ClipForm() {
		t.Fatal("tensor form read as a clip form")
	}
	if err := ValidateReferenceEditRequestForm(tensor); err != nil {
		t.Fatal(err)
	}
}

// TestSourceVideoFromGIFRoundTrips encodes a planar signed-unit clip through
// the production GIF encoder and decodes it back within the byte
// quantization, in the planar channel-major layout the edit consumes.
func TestSourceVideoFromGIFRoundTrips(t *testing.T) {
	const frames, height, width = 3, 4, 6
	encoder, err := NewGIFEncoder(16, SignedUnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	// Every pixel is a corner of the signed-unit cube, an exact palette
	// colour, so the dithered encoder reproduces it and the comparison pins
	// the channel, frame and position mapping rather than the palette.
	want := make([]float32, media.RGBChannels*frames*height*width)
	plane := height * width
	for frame := range frames {
		frameValues := make([]float32, media.RGBChannels*plane)
		for channel := range media.RGBChannels {
			for index := range plane {
				value := float32(-1)
				if (channel == frame%media.RGBChannels) != (index%width >= width/2) {
					value = 1
				}
				frameValues[channel*plane+index] = value
				want[(channel*frames+frame)*plane+index] = value
			}
		}
		if err := encoder.Add(frame, frameValues, height, width); err != nil {
			t.Fatal(err)
		}
	}
	video, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	source, err := SourceVideoFromGIF(video.Data)
	if err != nil {
		t.Fatal(err)
	}
	if source.Channels != media.RGBChannels || source.Frames != frames || source.Height != height || source.Width != width || len(source.Pixels) != len(want) {
		t.Fatalf("source shape = %dx%dx%dx%d pixels=%d", source.Channels, source.Frames, source.Height, source.Width, len(source.Pixels))
	}
	const tolerance = 0.02
	for index := range want {
		if math.Abs(float64(source.Pixels[index]-want[index])) > tolerance {
			t.Fatalf("pixel %d = %g want %g", index, source.Pixels[index], want[index])
		}
	}
	if media.U8ToSignedUnitF32(media.SignedUnitF32ToU8(0.25)) < 0.24 || media.U8ToSignedUnitF32(media.SignedUnitF32ToU8(0.25)) > 0.26 {
		t.Fatal("byte mapping does not invert")
	}
}
