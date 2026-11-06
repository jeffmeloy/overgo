package sensenovarecipe

import (
	"testing"

	"overgo/internal/media"
	"overgo/internal/routedlm"
	"overgo/internal/testcheck"
)

func TestDecodeGeneratedImagePublishesPlanarPNG(t *testing.T) {
	planar := []float32{
		-1, 1, 0, 0,
		0, 0, -1, 1,
		1, -1, 0, 0,
	}
	patches, height, width, err := media.PackPlanar(planar, 3, 2, 2, 2, media.PatchChannelsLast)
	testcheck.NoError(t, err)
	image := testcheck.Must(DecodeGeneratedImage(patches, routedlm.FlowPlan{VisionChannels: 3}, routedlm.FlowImagePlan{
		Width: 2, Height: 2, TokenWidth: width, TokenHeight: height, TokenPatch: 2,
	}))(t)
	if image.Width != 2 || image.Height != 2 || image.Channels != 3 || image.MediaType != "image/png" || len(image.Data) == 0 {
		t.Fatalf("image = %+v", image)
	}
}
