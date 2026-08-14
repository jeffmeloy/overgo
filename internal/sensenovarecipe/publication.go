package sensenovarecipe

import (
	"fmt"

	"overgo/internal/latentimage"
	"overgo/internal/routedlm"
)

// DecodeGeneratedImage publishes the terminal routed-image state.
func DecodeGeneratedImage(patches []float32, plan routedlm.FlowPlan, image routedlm.FlowImagePlan) (latentimage.EncodedImage, error) {
	if plan.VisionChannels != 3 {
		return latentimage.EncodedImage{}, fmt.Errorf("sensenova recipe: output channels=%d, want 3", plan.VisionChannels)
	}
	planar, err := latentimage.UnpackPlanarF32(
		patches, plan.VisionChannels, image.TokenHeight, image.TokenWidth,
		image.TokenPatch, latentimage.PatchChannelsLast,
	)
	if err != nil {
		return latentimage.EncodedImage{}, err
	}
	return latentimage.EncodePlanarPNG(planar, image.Height, image.Width)
}
