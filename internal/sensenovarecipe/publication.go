// Package sensenovarecipe runs the SenseNova-U1-8B MoT image-generation recipe.
//   - the recipe = generation only (task image-gen), activated and resolved through internal/modelrecipe as every generation recipe is
//   - Generator answers one GenerationRequest from the routed model's resident state
//   - DecodeGeneratedImage publishes the terminal state as an encoded image
//   - every dimension is owned by a tensor or the configuration, read through internal/routedlm
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
