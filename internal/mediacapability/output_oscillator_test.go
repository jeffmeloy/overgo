package mediacapability

import (
	"bytes"
	"image"
	"image/color/palette"
	"image/gif"
	"testing"

	"overgo/internal/media"
	"overgo/internal/oscillatorimage"
)

// TestOutputContentPublishesOscillatorClip pins the oscillator clip's
// publication: the encoded video carries its frame rate, so the GIF
// content validation admits it, and a clip without a rate is refused
// rather than published as a malformed video.
func TestOutputContentPublishesOscillatorClip(t *testing.T) {
	frame := image.NewPaletted(image.Rect(0, 0, 2, 2), palette.Plan9)
	delay, err := media.GIFFrameDelay(8, 0)
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := gif.EncodeAll(&encoded, &gif.GIF{Image: []*image.Paletted{frame}, Delay: []int{delay}}); err != nil {
		t.Fatal(err)
	}
	clip := oscillatorimage.EncodedVideo{
		Data: encoded.Bytes(), MediaType: media.GIFMediaType, Frames: 1, Channels: media.RGBChannels, Height: 2, Width: 2, FPS: 8,
	}
	content, err := OutputContent(clip)
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.MediaType != media.GIFMediaType || len(content.Data) != encoded.Len() {
		t.Fatalf("published %s (%d bytes)", content.Descriptor.MediaType, len(content.Data))
	}
	clip.FPS = 0
	if _, err := OutputContent(clip); err == nil {
		t.Fatal("a clip without a frame rate was published")
	}
	var decoded oscillatorimage.EncodedVideo
	if err := decoded.DecodeContent(content); err != nil {
		t.Fatal(err)
	}
	if decoded.FPS != 8 || decoded.Frames != 1 || decoded.Width != 2 {
		t.Fatalf("decoded clip = %+v", decoded)
	}
}
