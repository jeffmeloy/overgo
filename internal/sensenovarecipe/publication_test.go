package sensenovarecipe

import (
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/routedlm"
)

func TestDecodeGeneratedImagePublishesPlanarPNG(t *testing.T) {
	planar := []float32{
		-1, 1, 0, 0,
		0, 0, -1, 1,
		1, -1, 0, 0,
	}
	patches, height, width, err := latentimage.PackPlanarF32(planar, 3, 2, 2, 2, latentimage.PatchChannelsLast)
	if err != nil {
		t.Fatal(err)
	}
	image, err := DecodeGeneratedImage(patches, routedlm.FlowPlan{VisionChannels: 3}, routedlm.FlowImagePlan{
		Width: 2, Height: 2, TokenWidth: width, TokenHeight: height, TokenPatch: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if image.Width != 2 || image.Height != 2 || image.Channels != 3 || image.MediaType != "image/png" || len(image.Data) == 0 {
		t.Fatalf("image = %+v", image)
	}
}
