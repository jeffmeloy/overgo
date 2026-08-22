//go:build windows

package routedlm

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/media"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
)

// SeededFlowLatent ports adaptive's generation-state initialization: a
// torch.randn planar image, BF16 storage, noise scaling, then channels-last
// patch packing. Geometry and scale come only from the compiled flow plan.
func SeededFlowLatent(stream *torchrng.Stream, state *device.State, plan FlowPlan, image FlowImagePlan) ([]float32, error) {
	if stream == nil {
		return nil, fmt.Errorf("routed lm seeded latent: nil torchrng stream")
	}
	if plan.VisionChannels <= 0 || image.Width <= 0 || image.Height <= 0 || image.TokenPatch <= 0 {
		return nil, fmt.Errorf("routed lm seeded latent: invalid plan geometry")
	}
	planar := make([]float32, plan.VisionChannels*image.Width*image.Height)
	if err := stream.FillHost(state, planar); err != nil {
		return nil, err
	}
	for i := range planar {
		planar[i] = dtype.RoundBF16(planar[i])
		planar[i] = dtype.RoundBF16(planar[i] * float32(image.NoiseScale))
	}
	patches, gh, gw, err := media.PackPlanar(
		planar, plan.VisionChannels, image.Height, image.Width, image.TokenPatch, media.PatchChannelsLast,
	)
	if err != nil {
		return nil, err
	}
	if gh != image.TokenHeight || gw != image.TokenWidth || len(patches) != image.Tokens*plan.FlowDim {
		return nil, fmt.Errorf("routed lm seeded latent: packed [%d,%d,%d], want [%d,%d,%d]",
			gh, gw, len(patches), image.TokenHeight, image.TokenWidth, image.Tokens*plan.FlowDim)
	}
	return patches, nil
}
