package sensenovarecipe

import (
	"fmt"

	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/routedlm"
)

// DecodeGeneratedImage publishes the terminal routed-image state.
func DecodeGeneratedImage(patches []float32, plan routedlm.FlowPlan, image routedlm.FlowImagePlan) (latentimage.EncodedImage, error) {
	if plan.VisionChannels != 3 {
		return latentimage.EncodedImage{}, fmt.Errorf("sensenova recipe: output channels=%d, want 3", plan.VisionChannels)
	}
	planar, err := media.UnpackPlanar(
		patches, plan.VisionChannels, image.TokenHeight, image.TokenWidth,
		image.TokenPatch, media.PatchChannelsLast,
	)
	if err != nil {
		return latentimage.EncodedImage{}, err
	}
	return latentimage.EncodePlanarPNG(planar, image.Height, image.Width)
}
