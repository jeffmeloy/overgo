package latentvideo

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/gif"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
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
	if _, err := resolveEditPolicy(edit); err == nil {
		t.Fatal("a directory without the checkpoint resolved a policy")
	}
	wanDirectory, checkpoint := LiveEditArtifacts(edit)
	if wanDirectory != filepath.Join(models, "Wan2.1-T2V-1.3B") || checkpoint != filepath.Join(edit, "ar-forcing_002000.pt") {
		t.Fatalf("artifacts = %s, %s", wanDirectory, checkpoint)
	}
	if err := os.WriteFile(checkpoint, []byte("checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	policy, err := resolveEditPolicy(edit)
	if err != nil {
		t.Fatal(err)
	}
	if policy.FramesPerChunk != 3 || policy.LocalAttentionFrames != 21 || policy.InferenceSteps != 1000 ||
		policy.Shift != 5 || policy.SigmaMin != 0 || policy.SigmaMax != 1 || !policy.ExtraStep || policy.ContextTimestep != 0 ||
		len(policy.DenoisingTimesteps) != 4 || policy.DenoisingTimesteps[0] != 1000 || policy.DenoisingTimesteps[3] != 250 {
		t.Fatalf("policy = %+v", policy)
	}
	// The upstream latent of 60 by 104 at the autoencoder's stride of eight:
	// the size a clip is cut to for this editor.
	if width, height, err := EditSourceSize(edit); err != nil || width != 832 || height != 480 {
		t.Fatalf("edit source size = %dx%d, %v, want 832x480", width, height, err)
	}
	sizeless := policy
	sizeless.SourceWidth = 0
	if err := sizeless.validate(); err == nil {
		t.Fatal("a policy that declares no source size was accepted")
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
	clipID, err := artifact.ParseID("file:sha256:" + strings.Repeat("ab", 32))
	if err != nil {
		t.Fatal(err)
	}
	clip := ReferenceEditRequest{Prompt: "a fox", Seed: 7, SourceArtifact: clipID}
	if !clip.ClipForm() {
		t.Fatal("clip form not recognised")
	}
	if err := validateReferenceEditRequestForm(clip); err != nil {
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
		"empty":         {},
		"prompt only":   {Prompt: "a fox"},
		"artifact only": {SourceArtifact: clip.SourceArtifact},
		"negative seed": {Prompt: "a fox", Seed: -1, SourceArtifact: clip.SourceArtifact},
	} {
		if validateReferenceEditRequestForm(request) == nil {
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
	if err := validateReferenceEditRequestForm(tensor); err != nil {
		t.Fatal(err)
	}
}

// TestSourceFramesKeepEveryPixel holds the form a source clip is carried in
// without loss: a clip of values a palette could not hold, a different one at
// every pixel, comes back as it went, to the byte it was written at, with no
// column taking any of its neighbour's as a GIF's smoothing gives it. The same
// clip through a GIF comes back changed, which is what the form is for. A clip
// that is not three colours a pixel, holds a value that is no number, or is
// not the size it says is refused, and so are bytes that are no frames, frames
// of two sizes, and a clip of a form the editor does not read.
func TestSourceFramesKeepEveryPixel(t *testing.T) {
	const frames, height, width = 3, 4, 6
	plane := height * width
	clip := SourceVideo{Pixels: make([]float32, media.RGBChannels*frames*plane), Channels: media.RGBChannels, Frames: frames, Height: height, Width: width}
	for index := range clip.Pixels {
		// A value at every pixel that is its neighbour's by a different step in each channel and frame.
		clip.Pixels[index] = media.U8ToUnitF32(uint8((index*37 + index/plane*11) % 256))
	}
	content, err := SourceFramesContent(clip)
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.MediaType != media.PNGFramesMediaType || content.Descriptor.ID.Kind() != artifact.KindFile {
		t.Fatalf("source frames are published as %s %s", content.Descriptor.ID.Kind(), content.Descriptor.MediaType)
	}
	read, err := sourceVideoFromContent(t.Context(), content, editPolicy{}, Profile{})
	if err != nil {
		t.Fatal(err)
	}
	if read.Channels != media.RGBChannels || read.Frames != frames || read.Height != height || read.Width != width || len(read.Pixels) != len(clip.Pixels) {
		t.Fatalf("source shape = %dx%dx%dx%d pixels=%d", read.Channels, read.Frames, read.Height, read.Width, len(read.Pixels))
	}
	for index, want := range clip.Pixels {
		if read.Pixels[index] != want {
			t.Fatalf("pixel %d = %g, sent as %g", index, read.Pixels[index], want)
		}
	}

	// The same clip as a GIF: the form the frames replace does not keep it.
	encoder, err := NewClipEncoder(16, UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	for frame := range frames {
		values := make([]float32, media.RGBChannels*plane)
		for channel := range media.RGBChannels {
			copy(values[channel*plane:][:plane], clip.Pixels[(channel*frames+frame)*plane:][:plane])
		}
		if err := encoder.Add(frame, values, height, width); err != nil {
			t.Fatal(err)
		}
	}
	video, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	gifContent, err := VideoContent(video)
	if err != nil {
		t.Fatal(err)
	}
	smoothed, err := sourceVideoFromContent(t.Context(), gifContent, editPolicy{}, Profile{})
	if err != nil {
		t.Fatal(err)
	}
	var moved int
	for index, want := range clip.Pixels {
		if smoothed.Pixels[index] != want {
			moved++
		}
	}
	if moved == 0 {
		t.Fatal("a GIF kept every pixel of a clip no palette holds: the test's clip tells nothing")
	}

	refused := map[string]SourceVideo{
		"one colour a pixel": {Pixels: make([]float32, frames*plane), Channels: 1, Frames: frames, Height: height, Width: width},
		"a short clip":       {Pixels: clip.Pixels[1:], Channels: media.RGBChannels, Frames: frames, Height: height, Width: width},
		"no frames":          {Channels: media.RGBChannels, Height: height, Width: width},
	}
	unnumbered := SourceVideo{Pixels: append([]float32(nil), clip.Pixels...), Channels: media.RGBChannels, Frames: frames, Height: height, Width: width}
	unnumbered.Pixels[plane] = float32(math.NaN())
	refused["a value that is no number"] = unnumbered
	for name, source := range refused {
		if _, err := SourceFramesContent(source); err == nil {
			t.Errorf("%s: published as source frames", name)
		}
	}

	small, err := SourceFramesContent(SourceVideo{Pixels: make([]float32, media.RGBChannels), Channels: media.RGBChannels, Frames: 1, Height: 1, Width: 1})
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{
		"no bytes":            nil,
		"bytes of no picture": []byte("not a frame"),
		"a frame cut short":   content.Data[:len(content.Data)-1],
		"frames of two sizes": append(append([]byte(nil), content.Data...), small.Data...),
	} {
		if _, err := sourceVideoFromFrames(data); err == nil {
			t.Errorf("%s: read as a source clip", name)
		}
	}
	other := content
	other.Descriptor.MediaType = "video/webm"
	if _, err := sourceVideoFromContent(t.Context(), other, editPolicy{}, Profile{}); err == nil || !strings.Contains(err.Error(), "video/webm") {
		t.Fatalf("a clip of a form the editor does not read: %v", err)
	}
}

// TestSourceVideoFromGIFRoundTrips encodes a planar clip of values from 0 to
// 1 through the production GIF encoder and decodes it back within the byte
// quantization, in the planar channel-major layout and the range the edit
// consumes: a black pixel is 0 and not -1. The decoded clip is smoothed over
// a dither's reach, so the two columns at an edge each take a quarter of the
// other side and every other pixel keeps its value.
func TestSourceVideoFromGIFRoundTrips(t *testing.T) {
	const frames, height, width = 3, 4, 6
	encoder, err := NewGIFEncoder(16, UnitPixels)
	if err != nil {
		t.Fatal(err)
	}
	// Every pixel is a corner of the unit cube, an exact palette
	// colour, so the dithered encoder reproduces it and the comparison pins
	// the channel, frame and position mapping rather than the palette.
	want := make([]float32, media.RGBChannels*frames*height*width)
	plane := height * width
	for frame := range frames {
		frameValues := make([]float32, media.RGBChannels*plane)
		for channel := range media.RGBChannels {
			for index := range plane {
				value := float32(0)
				if (channel == frame%media.RGBChannels) != (index%width >= width/2) {
					value = 1
				}
				frameValues[channel*plane+index] = value
				// A column beside the edge is three parts its own side and one part the other.
				if column := index % width; column == width/2-1 || column == width/2 {
					value = 0.75*value + 0.25*(1-value)
				}
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
	source, err := sourceVideoFromGIF(video.Data)
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
	if media.U8ToUnitF32(media.UnitF32ToU8(0.25)) < 0.24 || media.U8ToUnitF32(media.UnitF32ToU8(0.25)) > 0.26 {
		t.Fatal("byte mapping does not invert")
	}
}

// TestSourceVideoFromGIFDisposesItsPartialFrames holds a source clip that
// arrives as a GIF whose frames each cover part of the picture, as a GIF from
// anywhere but this program's own encoder may: every frame is read as the
// frames say, the part a frame says to clear cleared and the part it
// says to put back put back before the next is drawn, and no frame is left
// out. The clip is two pixels wide and one high. Red fills it; green is drawn
// on the left and cleared after; blue is drawn on the right and taken back
// after; green is drawn on the left again. A reader that only draws each
// frame over the last keeps the green in the third frame and the blue in the
// fourth. The values wanted are those four pictures smoothed as a GIF source
// is, each pixel three parts itself and one part its neighbour. Bytes that
// are no GIF are refused.
func TestSourceVideoFromGIFDisposesItsPartialFrames(t *testing.T) {
	const frames, height, width = 4, 1, 2
	palette := color.Palette{color.RGBA{}, color.RGBA{R: 255, A: 255}, color.RGBA{G: 255, A: 255}, color.RGBA{B: 255, A: 255}}
	part := func(left, right int, colours ...uint8) *image.Paletted {
		picture := image.NewPaletted(image.Rect(left, 0, right, height), palette)
		copy(picture.Pix, colours)
		return picture
	}
	var encoded bytes.Buffer
	if err := gif.EncodeAll(&encoded, &gif.GIF{
		Image:    []*image.Paletted{part(0, 2, 1, 1), part(0, 1, 2), part(1, 2, 3), part(0, 1, 2)},
		Delay:    []int{1, 1, 1, 1},
		Disposal: []byte{gif.DisposalNone, gif.DisposalBackground, gif.DisposalPrevious, gif.DisposalNone},
		Config:   image.Config{ColorModel: palette, Width: width, Height: height},
	}); err != nil {
		t.Fatal(err)
	}
	source, err := sourceVideoFromGIF(encoded.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	// Each row is a colour and each pair in it a frame: red and red, green and
	// red, black and blue, green and red.
	want := []float32{
		1, 1, 0.25, 0.75, 0, 0, 0.25, 0.75,
		0, 0, 0.75, 0.25, 0, 0, 0.75, 0.25,
		0, 0, 0, 0, 0.25, 0.75, 0, 0,
	}
	if source.Channels != media.RGBChannels || source.Frames != frames || source.Height != height || source.Width != width || len(source.Pixels) != len(want) {
		t.Fatalf("source shape = %dx%dx%dx%d pixels=%d", source.Channels, source.Frames, source.Height, source.Width, len(source.Pixels))
	}
	for index := range want {
		if source.Pixels[index] != want[index] {
			t.Fatalf("value %d (colour %d, frame %d, column %d) = %g, want %g\n got %v\nwant %v",
				index, index/(frames*width), index/width%frames, index%width, source.Pixels[index], want[index], source.Pixels, want)
		}
	}
	if refused, err := sourceVideoFromGIF([]byte("not a GIF")); err == nil {
		t.Fatalf("bytes that are no GIF were read as a source clip of %d frames", refused.Frames)
	}
}
